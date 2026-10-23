"""Recette locale réutilisable : tests et preuves disponibles, sans entraîner un modèle."""
from datetime import datetime, timezone
import io
import hashlib
import json
from pathlib import Path
import sys
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maintidoc.adaptation import validate_dataset
from maintidoc.reranker import verify_base


def main():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    print(stream.getvalue())
    checks = {"unit_tests": result.wasSuccessful()}
    evidence = {}
    for name in ("v1_live", "v2_photo_live", "v2_clip_live", "v2_multimodal_live"):
        path = ROOT / "data/evaluation" / f"{name}.json"
        checks[name] = path.exists() and json.loads(path.read_text(encoding="utf-8")).get("all_passed") is True
        if path.exists():
            evidence[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    demo = json.loads((ROOT / "data/evaluation/v2_reranker_demo.json").read_text(encoding="utf-8"))
    adapter = ROOT / demo["adapter"]
    reload_path = adapter / "evaluation_reloaded.json"
    checks["adaptation_weights_changed"] = demo["training"]["weights_changed"] is True
    checks["three_way_comparison_reloaded"] = reload_path.exists() and (
        json.loads(reload_path.read_text(encoding="utf-8"))["test"]["means"] == demo["test"]["means"])
    try:
        verify_base(ROOT)
        checks["initial_model_hash"] = True
    except (OSError, ValueError):
        checks["initial_model_hash"] = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            checks["api_v2_running"] = json.load(response).get("version") == "2.0.0"
        with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=5) as response:
            checks["ui_running"] = response.status == 200
    except OSError:
        checks["services_reachable"] = False
    business_file = ROOT / "data/adaptation/metier.json"
    business = {"valid_dataset_available": False, "verified_pairs": 0, "acceptance": "requires_human_domain_review"}
    if business_file.exists():
        try:
            validation = validate_dataset(json.loads(business_file.read_text(encoding="utf-8")), root=ROOT)
            business.update(valid_dataset_available=True, verified_pairs=validation["pairs"])
        except (OSError, ValueError) as error:
            business["dataset_error"] = str(error)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "unit_tests_run": result.testsRun,
              "checks": checks, "evidence": evidence,
              "technical_checks_passed": all(checks.values()), "business": business,
              "notice": "Les recettes VLM/CLIP et l'adaptation utilisent des fixtures synthétiques. La présence de rapports réussis ne certifie pas une précision métier."}
    destination = ROOT / "data/evaluation/v2_final_checks.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["technical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
