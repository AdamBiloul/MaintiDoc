"""Reranking local, après filtrage des droits ; processus éphémère CPU."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def verify_base(root):
    root = Path(root)
    record = json.loads((root / "data/evaluation/reranker_model.json").read_text(encoding="utf-8"))
    directory = root / ".models/reranker-base"
    weights = directory / "model.safetensors"
    if not weights.exists():
        weights = directory / "pytorch_model.bin"
    with weights.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != record["weights_sha256"]:
        raise ValueError("Les poids du reranker initial ont changé.")
    return record


class LocalReranker:
    def __init__(self, root, mode="initial", adapter=None, *, evaluation_demo=False):
        self.root, self.mode = Path(root), mode
        if mode not in {"off", "initial", "adapte"}:
            raise ValueError("Mode de reranker inconnu.")
        self.adapter = Path(adapter).resolve() if adapter else None
        if mode == "adapte":
            if self.adapter is None or not self.adapter.is_relative_to((self.root / ".models").resolve()):
                raise ValueError("Adaptateur local autorisé requis.")
            metadata = json.loads((self.adapter / "adaptation.json").read_text(encoding="utf-8"))
            if metadata.get("demo_only") is not False and not evaluation_demo:
                raise ValueError("Un adaptateur de démonstration ne peut pas être activé dans l'application.")
            weights = self.adapter / "adapter.safetensors"
            if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["adapter_sha256"]:
                raise ValueError("Empreinte de l'adaptateur invalide.")
            base = json.loads((self.root / "data/evaluation/reranker_model.json").read_text(encoding="utf-8"))
            if metadata["base"] != base:
                raise ValueError("La révision du reranker initial a changé.")

    def scores(self, question, passages):
        if not passages:
            return []
        if self.mode == "off":
            raise ValueError("Le mode off ne produit pas de scores.")
        base = self.root / ".models/reranker-base"
        if not (base / "config.json").is_file():
            raise RuntimeError("Reranker non installé : scripts/download_reranker.py.")
        request = {"query": question, "passages": passages, "adapter": str(self.adapter) if self.adapter else None}
        try:
            result = subprocess.run([sys.executable, str(self.root / "scripts/reranker_worker.py"), str(base)],
                                    input=json.dumps(request, ensure_ascii=True), text=True, encoding="utf-8",
                                    capture_output=True, timeout=240, cwd=self.root,
                                    env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                                             PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Délai du reranker dépassé.")
        if result.returncode:
            raise RuntimeError("Échec du reranker local.")
        try:
            scores = json.loads(result.stdout)["scores"]
            if len(scores) != len(passages) or not all(type(s) in (int, float) and math.isfinite(s) for s in scores):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise RuntimeError("Scores du reranker invalides.")
        return scores

    def rerank(self, question, hits):
        if self.mode == "off" or not hits:
            return hits
        scores = self.scores(question, [h["text"] for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        # Le score BGE conservé sert au seuil documentaire ; le logit n'est pas une confiance.
        return [dict(hits[i], rerank_score=float(scores[i]), rerank_position=rank)
                for rank, i in enumerate(order)]
