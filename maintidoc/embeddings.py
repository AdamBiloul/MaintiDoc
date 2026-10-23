import hashlib
import json
from pathlib import Path
import threading
import os
import subprocess
import sys
import numpy as np


class BGE:
    name = "BAAI/bge-m3"

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / ".models/bge-m3"
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            # Le processus se termine AVANT la génération : Windows récupère toute
            # la mémoire du modèle et de PyTorch, même celle de leurs allocateurs.
            process = subprocess.run(
                [sys.executable, str(self.root / "scripts/encode_worker.py"), str(self.path)],
                input=json.dumps(texts, ensure_ascii=True), text=True, encoding="utf-8",
                capture_output=True, timeout=180, cwd=self.root,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                         PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if process.returncode:
                raise RuntimeError("Le calcul local des embeddings a échoué ; vérifiez la mémoire disponible et les poids BGE-M3.")
            vectors = np.asarray(json.loads(process.stdout), dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def corpus_vectors(self, chunks):
        texts = [c["text"] for c in chunks]
        key = hashlib.sha256(json.dumps([str(self.path), texts], ensure_ascii=False).encode()).hexdigest()
        destination = self.root / "data/embeddings" / f"v1_{key}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                return saved["vectors"]
        # Réutilise les vecteurs des séances précédentes si les passages sont identiques.
        old_source = self.root / "data/chunks/feuille_de_route.json"
        vectors = None
        if old_source.exists():
            raw = old_source.read_bytes()
            old = json.loads(raw)
            old_key = hashlib.sha256(raw + str(self.path.resolve()).encode()).hexdigest()
            old_cache = self.root / "data/embeddings" / f"{old_key}.npz"
            if [c["id"] for c in old] == [c["id"] for c in chunks] and old_cache.exists():
                with np.load(old_cache, allow_pickle=False) as saved:
                    vectors = saved["vectors"]
        if vectors is None:
            vectors = self.encode(texts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, vectors=vectors)
        return vectors
