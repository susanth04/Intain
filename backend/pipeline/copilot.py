"""
copilot.py - Stage 8: AI Reviewer
Live LLM answers grounded in loan model outputs.
"""

import os
import time
import json
import traceback
from models.loader import PROJECT_ROOT
from env_loader import load_env

load_env()

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.copilot.rag_grounding import RAGGrounder
    from src.copilot.reviewer_copilot import ReviewerCopilot
    import yaml

    with open(PROJECT_ROOT / "config.yaml") as f:
        _CFG = yaml.safe_load(f)

    _RAG = RAGGrounder(_CFG)
    _COPILOT = ReviewerCopilot(_CFG, _RAG)
    COPILOT_AVAILABLE = True
except Exception as e:
    COPILOT_AVAILABLE = False
    COPILOT_ERROR = str(e)

_CONTEXT_CACHE = {}


def _num(v, digits=4):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:
            return None
        return round(f, digits)
    except Exception:
        return None


def _compact_facts(loan_id: str) -> dict:
    cached = _CONTEXT_CACHE.get(loan_id)
    if cached:
        return cached

    import models.loader as loader
    import pipeline.prediction as prediction

    facts = {"loan_id": loan_id}
    try:
        df = loader._DATA["panel"]
        rows = df[df["loan_id"] == loan_id]
        if not rows.empty:
            sort_col = "month_index" if "month_index" in rows.columns else "period"
            row = rows.sort_values(sort_col).iloc[-1]
            keep = [
                "loan_id", "month_index", "current_status", "days_past_due", "credit_score_band",
                "ltv_band", "dti_band", "current_balance", "original_balance", "interest_rate",
                "state", "servicer_name", "loan_age_months", "default_flag", "prepayment_flag",
                "document_status",
            ]
            facts["loan"] = {k: (None if (isinstance(row.get(k), float) and row.get(k) != row.get(k)) else row.get(k)) for k in keep if k in row.index}
        else:
            facts["loan"] = {"error": "not_found"}
    except Exception as e:
        facts["loan"] = {"error": str(e)}

    try:
        pred = prediction.run(loan_id)
        slim_preds = {}
        for key, block in (pred.get("predictions") or {}).items():
            if isinstance(block, dict):
                slim_preds[key] = {
                    "probability": block.get("probability"),
                    "risk_tier": block.get("risk_tier"),
                }
        facts["predictions"] = slim_preds
        ns = pred.get("next_state") or {}
        facts["next_state"] = {
            "predicted_state": ns.get("predicted_state"),
            "class_probabilities": ns.get("class_probabilities"),
        }
        facts["recommended_action"] = pred.get("recommended_action")
        facts["ensemble_confidence"] = pred.get("ensemble_confidence")
    except Exception as e:
        facts["predictions_error"] = str(e)

    try:
        anom_df = loader._DATA.get("anomaly_scores")
        if anom_df is not None and not anom_df.empty:
            hit = anom_df[anom_df["loan_id"] == loan_id]
            if not hit.empty:
                sort_col = "month_index" if "month_index" in hit.columns else hit.columns[0]
                latest = hit.sort_values(sort_col).iloc[-1]
                facts["anomaly"] = {
                    "anomaly_score": _num(latest.get("anomaly_score")),
                    "exception_flag": int(latest.get("exception_flag") or 0),
                    "exception_type": latest.get("predicted_exception_type") or "",
                    "top_drivers": latest.get("top_drivers") or "",
                }
            else:
                facts["anomaly"] = {"anomaly_score": 0, "note": "no precomputed score"}
        else:
            facts["anomaly"] = {"note": "anomaly table not loaded"}
    except Exception as e:
        facts["anomaly_error"] = str(e)

    _CONTEXT_CACHE[loan_id] = facts
    if len(_CONTEXT_CACHE) > 64:
        _CONTEXT_CACHE.pop(next(iter(_CONTEXT_CACHE)))
    return facts


SYSTEM_PROMPT = """You are the Intain reviewer copilot, a mortgage credit analyst.
Answer the reviewer's question using ONLY the Live Model Data JSON.
Use exact numbers from that JSON. Do not invent fields, balances, or dates.
Write a natural analyst note in markdown with:
## Verdict
## Drivers
## Watchouts
## Recommendation
Keep it under 220 words. Every claim must be traceable to the JSON."""


def _call_llm(api_key: str, base_url: str, model_name: str, user_message: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=45.0,
    )
    llm_resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    choice = llm_resp.choices[0].message
    content = (choice.content or "").strip()
    if content:
        return content
    raise RuntimeError("LLM returned an empty response. Check model name and API key.")


def run(loan_id, question=None):
    t0 = time.perf_counter()
    retrieved = []

    if not COPILOT_AVAILABLE:
        return {
            "loan_id": loan_id,
            "status": "error",
            "error": "Copilot modules could not be initialized",
            "details": getattr(sys.modules[__name__], "COPILOT_ERROR", "Unknown error"),
            "answer": "Error: Copilot subsystem unavailable."
        }

    load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model_name = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    if not api_key:
        return {
            "loan_id": loan_id,
            "status": "error",
            "offline_mode": True,
            "error": "OPENAI_API_KEY is not loaded. Put it in backend/.env and restart the API.",
            "answer": "LLM is not configured. Add OPENAI_API_KEY to backend/.env, restart the backend, then ask again.",
        }

    try:
        facts = _compact_facts(loan_id)
        loan_context = json.dumps(facts, indent=2, default=str)
        effective_question = question or ("Give a reviewer brief for loan " + loan_id)
        retrieved = _RAG.retrieve(effective_question, top_k=2)
        rag_context = _RAG.format_context(retrieved)
        if len(rag_context) > 1200:
            rag_context = rag_context[:1200] + "\n[truncated]"

        user_message = (
            "## Dictionary snippets\n" + rag_context + "\n\n"
            "## Live Model Data\n" + loan_context + "\n\n"
            "## Question\n" + effective_question
        )
        response = _call_llm(api_key, base_url, model_name, user_message)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "loan_id": loan_id,
            "question": question,
            "answer": response,
            "offline_mode": False,
            "model": model_name,
            "rag_docs_retrieved": len(retrieved),
            "execution_ms": elapsed_ms,
            "disclaimer": "AI-generated recommendation — not a decision. Human review required.",
        }

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        err = str(e)
        return {
            "loan_id": loan_id,
            "status": "error",
            "offline_mode": False,
            "error": err,
            "traceback": traceback.format_exc(),
            "answer": "LLM call failed: " + err,
            "execution_ms": elapsed_ms,
        }
