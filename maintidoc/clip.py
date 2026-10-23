"""Encodeurs texte multilingue et image dans le même espace CLIP 512D."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


class ClipEncoder:
    dimensions = 512

    def __init__(self, root):
        self.root = Path(root)

    @property
    def signature(self):
        manifest = self.root / "data/evaluation/clip_models.json"
        if not manifest.exists():
            return "not-installed"
        return hashlib.sha256(manifest.read_bytes()).hexdigest()

    def _encode(self, kind, items):
        if self.signature == "not-installed":
            raise RuntimeError("Installez les encodeurs avec scripts/download_clip.py.")
        try:
            completed = subprocess.run(
                [sys.executable, str(self.root / "scripts/clip_worker.py"), kind,
                 str(self.root / ".models" / ("clip-text" if kind == "text" else "clip-image"))],
                input=json.dumps(items, ensure_ascii=True), text=True, encoding="utf-8",
                capture_output=True, timeout=240, cwd=self.root,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                         PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Le calcul CLIP a dépassé le délai autorisé.")
        if completed.returncode:
            raise RuntimeError("Échec de l'encodeur CLIP local ; vérifiez ses modèles et la mémoire disponible.")
        try:
            vectors = np.asarray(json.loads(completed.stdout), dtype=np.float32)
        except (ValueError, TypeError):
            raise RuntimeError("Sortie CLIP illisible.")
        if vectors.shape != (len(items), self.dimensions) or not np.isfinite(vectors).all():
            raise RuntimeError("Sortie CLIP invalide.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("Vecteur CLIP nul.")
        return vectors / norms

    def text(self, texts):
        return self._encode("text", texts)

    def images(self, images):
        return self._encode("image", [base64.b64encode(image).decode("ascii") for image in images])
