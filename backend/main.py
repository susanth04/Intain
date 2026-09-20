"""
main.py — FastAPI Application
Production inference backend for Intain Loan Performance Intelligence Engine.
"""

import time
import os

from env_loader import load_env
load_env()

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from models.loader import _DATA, MODELS
import pipeline.ingestion as ingestion
import pipeline.data_quality as data_quality
import pipeline.features as features
import pipeline.prediction as prediction
import pipeline.anomaly as anomaly
import pipeline.explainability as explainability
import pipeline.scenario as scenario
import pipeline.copilot as copilot
import pipeline.survival as survival
import pipeline.portfolio as portfolio

app = FastAPI(
    title="Intain Inference API",
    description="Production backend for Loan Performance Intelligence Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRunRequest(BaseModel):
    loan_id: str
    stage: Optional[str] = None


class ScenarioRequest(BaseModel):
    loan_id: str
    scenario: str


class CopilotRequest(BaseModel):
    loan_id: str
    question: str


@app.get("/api/health")
def health_check():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "status": "ok",
        "models_loaded": len(MODELS),
        "data_panel_rows": len(_DATA.get("panel", [])),
        "llm_configured": bool(api_key),
        "llm_model": os.environ.get("OPENAI_MODEL", "gemini-2.5-flash"),
    }


@app.get("/api/loans")
def list_loans():
    panel = _DATA.get("panel")
    if panel is None or panel.empty:
        return {"loans": []}
    unique_loans = panel["loan_id"].dropna().unique().tolist()
    return {"loans": unique_loans[:1000], "total_count": len(unique_loans)}


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRunRequest):
    """
    Execute the inference pipeline for a given loan_id.
    """
    loan_id = request.loan_id
    panel = _DATA.get("panel")
    
    if panel is None or panel.empty:
        raise HTTPException(status_code=500, detail="Data panel not loaded")
        
    loan_records = panel[panel["loan_id"] == loan_id].copy()
    if loan_records.empty:
        raise HTTPException(status_code=404, detail=f"Loan {loan_id} not found")

    stages_results = []
    
    # Define execution order
    target_stages = [
        ("ingestion",      lambda: ingestion.run(loan_id)),
        ("quality",        lambda: data_quality.run(loan_id, loan_records)),
        ("features",       lambda: features.run(loan_id)),
        ("prediction",     lambda: prediction.run(loan_id)),
        ("anomaly",        lambda: anomaly.run(loan_id)),
        ("explainability", lambda: explainability.run(loan_id)),
        ("scenarios",      lambda: scenario.run(loan_id)),
        ("reviewer",       lambda: {"status": "waiting", "message": "Call /api/copilot directly"})
    ]

    # If specific stage requested, only run that and return immediately
    if request.stage:
        for stage_key, stage_func in target_stages:
            if stage_key == request.stage:
                try:
                    result = stage_func()
                    return {
                        "loan_id": loan_id,
                        "stages": [{
                            "key": stage_key,
                            "status": "completed",
                            "output": result,
                            "executionMs": result.get("execution_ms", 0)
                        }]
                    }
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    return {
                        "loan_id": loan_id,
                        "stages": [{
                            "key": stage_key,
                            "status": "failed",
                            "error": str(e)
                        }]
                    }
        raise HTTPException(status_code=400, detail=f"Unknown stage: {request.stage}")

    # Run full pipeline sequentially
    for stage_key, stage_func in target_stages:
        if stage_key == "reviewer":
            stages_results.append({
                "key": stage_key,
                "status": "completed",
                "output": {"status": "ready"}
            })
            continue
            
        try:
            result = stage_func()
            status = "completed"
            error = None
            if isinstance(result, dict) and result.get("status") == "error":
                status = "failed"
                error = result.get("error", "Unknown error")
                
            stages_results.append({
                "key": stage_key,
                "status": status,
                "output": result,
                "executionMs": result.get("execution_ms", 0),
                "error": error
            })
            
            if status == "failed" and stage_key in ["ingestion", "features"]:
                # Stop pipeline if critical stage fails
                break
                
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            stages_results.append({
                "key": stage_key,
                "status": "failed",
                "error": str(e)
            })
            if stage_key in ["ingestion", "features"]:
                break

    return {
        "loan_id": loan_id,
        "stages": stages_results
    }


@app.post("/api/scenario")
def run_scenario(request: ScenarioRequest):
    try:
        return scenario.run(request.loan_id, request.scenario)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/copilot")
def ask_copilot(request: CopilotRequest):
    try:
        return copilot.run(request.loan_id, request.question)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


class PortfolioRequest(BaseModel):
    limit: Optional[int] = 24


@app.get("/api/portfolio/overview")
def portfolio_overview(limit: int = 24):
    """Judge-demo aggregates: heatmaps, risk tiers, honest metrics."""
    try:
        return portfolio.overview(limit)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/heatmap")
def portfolio_heatmap(limit: int = 24):
    try:
        data = portfolio.overview(limit)
        return {
            "status_heatmap": data.get("status_heatmap"),
            "dpd_heatmap": data.get("dpd_heatmap"),
            "scenario_heatmap": data.get("scenario_heatmap"),
            "state_heatmap": data.get("state_heatmap"),
            "tier_summary": data.get("tier_summary"),
            "total_loans": data.get("total_loans"),
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
def model_metrics():
    try:
        return portfolio._load_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/survival/{loan_id}")
def get_survival(loan_id: str):
    """Return survival time estimate for a single loan."""
    try:
        return survival.run(loan_id)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

