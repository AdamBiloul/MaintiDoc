"""Recette CLIP réelle sur deux figures synthétiques, hors du corpus métier."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
import pymupdf
from qdrant_client import QdrantClient

from maintidoc.clip import ClipEncoder
from maintidoc.figures import build_figure_index
from maintidoc.settings import ROOT

started = time.perf_counter()
encoder = ClipEncoder(ROOT)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "data/raw").mkdir(parents=True)
    path = root / "data/raw/synthetic_figures.pdf"
    pdf = pymupdf.open()
    for shape, color in [("circle", "orange"), ("square", "blue")]:
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        if shape == "circle":
            draw.ellipse((40, 40, 184, 184), fill=color)
        else:
            draw.rectangle((40, 40, 184, 184), fill=color)
        stream = BytesIO()
        image.save(stream, format="PNG")
        page = pdf.new_page()
        page.insert_image(pymupdf.Rect(50, 50, 274, 274), stream=stream.getvalue())
        page.insert_text((50, 310), f"Synthetic fixture: {color} {shape}.")
    pdf.save(path)
    pdf.close()
    state = {"collection": "fixture", "documents": [{
        "id": "synthetic", "path": "data/raw/synthetic_figures.pdf", "version": "fixture-v1",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "allowed_roles": ["test"],
    }]}
    (root / "data/index_v1.json").write_text(json.dumps(state), encoding="utf-8")
    index = build_figure_index(root, encoder)
    assert index["count"] == 2
    print("Deux figures indexées avec CLIP réel.", flush=True)
    texts = ["un cercle orange", "un carré bleu"]
    text_vectors = encoder.text(texts)
    image_vector = encoder.images([(root / index["figures"][0]["image_path"]).read_bytes()])[0]
    client = QdrantClient(path=str(root / "data/qdrant_figures_v2"))
    try:
        cases = []
        for query, vector, expected in zip(texts, text_vectors, [1, 2]):
            hits = client.query_points(index["collection"], query=vector.tolist(), limit=2).points
            cases.append({"query": query, "expected_page": expected,
                          "retrieved_pages": [h.payload["page"] for h in hits],
                          "scores": [float(h.score) for h in hits], "passed": hits[0].payload["page"] == expected})
        image_hits = client.query_points(index["collection"], query=image_vector.tolist(), limit=2).points
        image_passed = image_hits[0].payload["page"] == 1
    finally:
        client.close()
report = {"fixture": "Deux figures géométriques synthétiques, pas des manuels métier",
          "dimensions": 512, "encoder_signature": encoder.signature, "text_cases": cases,
          "image_query_passed": image_passed, "seconds": round(time.perf_counter() - started, 2),
          "all_passed": all(c["passed"] for c in cases) and image_passed,
          "maintenance_accuracy_validated": False}
(ROOT / "data/evaluation/v2_clip_live.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True, indent=2))
if not report["all_passed"]:
    raise SystemExit("Un classement de la recette CLIP diffère de l'attendu : consultez le rapport.")
