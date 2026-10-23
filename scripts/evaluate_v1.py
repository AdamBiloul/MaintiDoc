"""Vérification réelle de l'API, de BGE-M3 et du modèle Ollama installé."""
import json
import argparse
from pathlib import Path
import time

import httpx

root = Path(__file__).resolve().parents[1]
token = (root / "data/runtime/demo-token.txt").read_text(encoding="utf-8").strip()
headers = {"Authorization": f"Bearer {token}"}
base = "http://127.0.0.1:8000"
cases = [
    ("Quels contrôles sont prévus avant la création d'un ticket ?", "answered", [3, 5], ["approbation"]),
    ("Quel modèle génère les embeddings et quelle base stocke les passages ?", "answered", [2], ["bge-m3", "qdrant"]),
    ("Quelles commandes du bot permettent d'approuver et d'annuler ?", "answered", [4], ["/approve", "/cancel"]),
    ("Quelle est la capitale de l'Australie ?", "abstained", [], []),
]
results = []
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-cases", type=int, default=4)
args = parser.parse_args()
with httpx.Client(timeout=240) as client:
    for question, expected, pages, evidence in cases[:args.max_cases]:
        started = time.perf_counter()
        try:
            response = client.post(f"{base}/ask", headers=headers, json={"question": question})
            response.raise_for_status()
            answer = response.json()
        except httpx.HTTPError as error:
            answer = {"status": "error", "citations": [], "error_type": type(error).__name__}
        valid = answer["status"] == expected
        if expected == "answered":
            valid = valid and bool(answer["citations"]) and any(c["page"] in pages for c in answer["citations"])
            proof = " ".join(c["quote"].lower() for c in answer["citations"])
            valid = valid and all(term in proof for term in evidence)
        results.append({"question": question, "expected_status": expected,
                        "passed": valid, "seconds": round(time.perf_counter() - started, 2), "response": answer})
        print(f"{answer['status']} ; vérification : {valid} ; {results[-1]['seconds']} s", flush=True)
        (root / "data/evaluation/v1_live_progress.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    denied = client.post(f"{base}/ask", json={"question": "Question sans identité"}).status_code
report = {"corpus": "Feuille de route uniquement", "model": "qwen2.5:1.5b", "results": results,
          "unauthenticated_status": denied, "all_passed": all(r["passed"] for r in results) and denied == 401}
(root / "data/evaluation/v1_live.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
if not report["all_passed"]:
    raise SystemExit("Certaines vérifications ont échoué : consultez data/evaluation/v1_live.json.")
