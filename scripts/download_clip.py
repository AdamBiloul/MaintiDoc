"""Télécharge les deux encodeurs CLIP compatibles, depuis leurs dépôts officiels."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(root / ".models/hf-home")
from huggingface_hub import HfApi, snapshot_download

api = HfApi()
records = []
for repository, directory in [
    ("sentence-transformers/clip-ViT-B-32", "clip-image"),
    ("sentence-transformers/clip-ViT-B-32-multilingual-v1", "clip-text"),
]:
    revision = api.model_info(repository).sha
    files = api.list_repo_files(repository, revision=revision)
    selected = []
    for name in files:
        if any(part in name.split("/") for part in ("onnx", "openvino", "tf", "flax")):
            continue
        if name.endswith((".json", ".txt", ".model", ".jinja", ".safetensors")):
            selected.append(name)
        elif name.endswith("pytorch_model.bin"):
            safe = name.removesuffix("pytorch_model.bin") + "model.safetensors"
            if safe not in files:
                selected.append(name)
    print(f"Téléchargement : {repository}", flush=True)
    snapshot_download(repository, revision=revision, local_dir=root / ".models" / directory,
                      allow_patterns=selected, max_workers=2)
    records.append({"repository": repository, "revision": revision, "directory": directory})
destination = root / "data/evaluation/clip_models.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps({"dimensions": 512, "precision": "float32", "models": records}, indent=2), encoding="utf-8")
print("Encodeurs CLIP téléchargés et révisions enregistrées.")
