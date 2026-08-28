"""
rag_grounding.py
================
Task 7 — RAG Grounding Layer

Provides a lightweight retrieval system over:
  1. data_dictionary.md  (field definitions)
  2. validation_rules.json (rule text)
  3. Model outputs (computed metrics, anomaly scores, scenario results)

Uses TF-IDF cosine similarity for retrieval — no external API needed.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


class RAGGrounder:
    """
    TF-IDF retrieval over the grounding corpus.
    All sources are loaded at init; retrieved at query time.
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.docs    = []   # list of {id, source, text}
        self.vectors = None
        self.tfidf   = None
        self._build_corpus()

    def _build_corpus(self):
        raw_dir  = ROOT / self.cfg["PATHS"]["raw_data"]
        proc_dir = ROOT / self.cfg["PATHS"]["processed_data"]

        # 1. Data dictionary
        dd_path = raw_dir / "data_dictionary.md"
        if dd_path.exists():
            text = dd_path.read_text(encoding="utf-8")
            # Split into table rows for fine-grained retrieval
            for line in text.splitlines():
                line = line.strip()
                if "|" in line and len(line) > 20:
                    self.docs.append({"id": f"dd_{len(self.docs)}", "source": "data_dictionary", "text": line})

        # 2. Validation rules
        rules_path = raw_dir / "validation_rules.json"
        if rules_path.exists():
            with open(rules_path) as f:
                rules = json.load(f).get("rules", [])
            for rule in rules:
                text = f"{rule['rule_id']}: {rule['name']}. {rule['description']}"
                self.docs.append({"id": rule["rule_id"], "source": "validation_rules", "text": text})

        # 3. Model metrics
        for fname in ["binary_model_metrics.json", "next_state_metrics.json", "survival_metrics.json"]:
            path = proc_dir / fname
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                text = f"Model metrics from {fname}: {json.dumps(data, indent=0)[:1000]}"
                self.docs.append({"id": fname, "source": "model_outputs", "text": text})

        # 4. Anomaly examples
        anomaly_path = ROOT / self.cfg["PATHS"]["reports"] / "anomaly_examples.md"
        if anomaly_path.exists():
            text = anomaly_path.read_text(encoding="utf-8")[:3000]
            self.docs.append({"id": "anomaly_examples", "source": "anomaly", "text": text})

        if not self.docs:
            # Fallback minimal doc
            self.docs.append({"id": "fallback", "source": "default",
                              "text": "Loan performance analysis system for mortgage portfolio monitoring."})

        # Fit TF-IDF
        corpus = [d["text"] for d in self.docs]
        self.tfidf   = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.vectors = self.tfidf.fit_transform(corpus)
        print(f"  [RAG] Corpus built: {len(self.docs)} documents indexed")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Return top_k most relevant documents for the query."""
        q_vec = self.tfidf.transform([query])
        sims  = cosine_similarity(q_vec, self.vectors)[0]
        top   = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.docs[i], "similarity": round(float(sims[i]), 4)}
            for i in top if sims[i] > 0.01
        ]

    def format_context(self, retrieved: list) -> str:
        """Format retrieved docs as a context string for the LLM prompt."""
        if not retrieved:
            return "No relevant context found."
        parts = []
        for doc in retrieved:
            parts.append(f"[Source: {doc['source']} | ID: {doc['id']} | sim={doc['similarity']:.3f}]\n{doc['text']}")
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    cfg = load_cfg()
    rag = RAGGrounder(cfg)
    results = rag.retrieve("what does days_past_due mean")
    for r in results:
        print(r["source"], "|", r["text"][:100])
