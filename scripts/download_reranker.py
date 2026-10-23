"""Téléchargement administratif d'un reranker multilingue, révision figée."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".models/hf-home")
from huggingface_hub import HfApi, snapshot_download

repository = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
api = HfApi()
revision = api.model_info(repository).sha
files = api.list_repo_files(repository, revision=revision)
selected = [f for f in files if "/" not in f and f.endswith((".json", ".txt", ".model", ".safetensors"))]
if "model.safetensors" not in files:
    selected.append("pytorch_model.bin")
destination = ROOT / ".models/reranker-base"
snapshot_download(repository, revision=revision, local_dir=destination,
                  allow_patterns=selected, max_workers=2)
weights = destination / ("model.safetensors" if "model.safetensors" in files else "pytorch_model.bin")
record = {"repository": repository, "revision": revision,
          "weights_sha256": hashlib.file_digest(weights.open("rb"), "sha256").hexdigest(),
          "max_length": 384, "precision": "float32"}
(ROOT / "data/evaluation/reranker_model.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print("Reranker multilingue téléchargé ; révision et empreinte enregistrées.")
