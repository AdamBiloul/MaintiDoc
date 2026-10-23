"""Compare un adaptateur sauvegardé au modèle initial et au classement sans reranker."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maintidoc.adaptation import compare_rankings, validate_dataset
from maintidoc.reranker import LocalReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    validation = validate_dataset(data, demo=args.demo, root=ROOT)
    adapted = LocalReranker(ROOT, "adapte", args.adapter, evaluation_demo=args.demo)
    metadata = json.loads((args.adapter / "adaptation.json").read_text(encoding="utf-8"))
    if metadata["dataset"]["dataset_sha256"] != validation["dataset_sha256"]:
        raise ValueError("Le jeu de données a changé depuis l'entraînement.")
    initial = LocalReranker(ROOT, "initial")
    report = compare_rankings([q for q in data["queries"] if q["split"] == "test"],
                              lambda mode, q, p: (initial if mode == "initial" else adapted).scores(q, p))
    destination = args.adapter / "evaluation_reloaded.json"
    destination.write_text(json.dumps({"demo_only": args.demo, "validation": validation, "test": report},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), "means": report["means"]}, indent=2))


if __name__ == "__main__":
    main()
