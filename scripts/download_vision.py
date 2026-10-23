"""Télécharge le petit VLM officiel pour le premier incrément de V2."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(root / ".models/hf-home")
from huggingface_hub import HfApi, snapshot_download

repository = "HuggingFaceTB/SmolVLM-256M-Instruct"
revision = HfApi().model_info(repository).sha
destination = root / ".models/smolvlm"
snapshot_download(repository, revision=revision, local_dir=destination,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                  ignore_patterns=["onnx/*", "original/*"], max_workers=2)
manifest = root / "data/evaluation/vision_model.json"
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps({"repository": repository, "revision": revision,
                               "language": "en", "local_path": ".models/smolvlm"}, indent=2), encoding="utf-8")
print("VLM téléchargé ; révision enregistrée dans data/evaluation/vision_model.json.")
