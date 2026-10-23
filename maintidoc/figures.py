"""Extraction de régions raster/vectorielles et index CLIP versionné."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf
from qdrant_client import QdrantClient

from maintidoc.clip import ClipEncoder
from maintidoc.settings import ROOT
from scripts.qdrant_store import index_chunks


def state_signature(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def extract_figures(root, text_state, max_per_page=100):
    root = Path(root)
    snapshot = state_signature(text_state)
    records = []
    destination = root / "data/figures_v2" / snapshot
    for document in text_state["documents"]:
        source = (root / document["path"]).resolve()
        if not source.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("PDF hors du corpus autorisé.")
        if hashlib.sha256(source.read_bytes()).hexdigest() != document["sha256"]:
            raise ValueError("Un PDF a changé : réindexez le corpus textuel avant les figures.")
        with pymupdf.open(source) as pdf:
            for page_number, page in enumerate(pdf, 1):
                original_rotation = page.rotation
                page.set_rotation(0)
                regions = [(pymupdf.Rect(info["bbox"]), "raster") for info in page.get_image_info()]
                regions.extend((pymupdf.Rect(rect), "vector_region") for rect in page.cluster_drawings())
                accepted = []
                for rect, kind in regions:
                    rect = rect & page.rect
                    if rect.is_empty or min(rect.width, rect.height) < 12 or rect.get_area() < 256:
                        continue
                    if kind == "vector_region" and rect.get_area() > 0.95 * page.rect.get_area():
                        continue
                    # Évite de dupliquer une même région détectée par plusieurs chemins.
                    if any((rect & old).get_area() / (rect | old).get_area() > 0.95 for old, _ in accepted):
                        continue
                    accepted.append((rect, kind))
                if len(accepted) > max_per_page:
                    raise ValueError("Trop de figures sur une page ; vérifiez le PDF avant indexation.")
                for rect, kind in accepted:
                    bbox = [round(float(value), 3) for value in rect]
                    identity = [snapshot, document["id"], page_number, bbox, kind]
                    identifier = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                    image_path = destination / f"{identifier}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    scale = min(2.0, 1024 / max(rect.width, rect.height))
                    png = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect,
                                          colorspace=pymupdf.csRGB, alpha=False).tobytes("png")
                    image_path.write_bytes(png)
                    records.append({
                        "id": identifier, "document_id": document["id"], "document": source.name,
                        "version": document["version"], "page": page_number, "bbox": bbox,
                        "kind": kind, "rotation_original": original_rotation,
                        "allowed_roles": document["allowed_roles"], "content_sha256": document["sha256"],
                        "image_sha256": hashlib.sha256(png).hexdigest(),
                        "image_path": str(image_path.relative_to(root)),
                        "text": " ".join(page.get_text("text", clip=rect + (-12, -12, 12, 12)).split()),
                    })
                page.set_rotation(original_rotation)
    return records


def build_figure_index(root=ROOT, encoder=None):
    root = Path(root)
    text_state = json.loads((root / "data/index_v1.json").read_text(encoding="utf-8"))
    figures = extract_figures(root, text_state)
    encoder = encoder or ClipEncoder(root)
    signature = encoder.signature
    key = hashlib.sha256(json.dumps([state_signature(text_state), signature, figures], sort_keys=True).encode()).hexdigest()
    collection = f"figures_{key}" if figures else None
    if figures:
        vectors = [None] * len(figures)
        pending = {}
        for i, figure in enumerate(figures):
            cache = root / "data/embeddings" / f"clip_{signature}_{figure['image_sha256']}.npy"
            if cache.exists():
                vectors[i] = np.load(cache, allow_pickle=False)
            else:
                item = pending.setdefault(figure["image_sha256"], {"cache": cache, "path": figure["image_path"], "positions": []})
                item["positions"].append(i)
        missing = list(pending.values())
        for start in range(0, len(missing), 4):
            batch = missing[start:start + 4]
            encoded = encoder.images([(root / item["path"]).read_bytes() for item in batch])
            for item, vector in zip(batch, encoded):
                item["cache"].parent.mkdir(parents=True, exist_ok=True)
                np.save(item["cache"], vector, allow_pickle=False)
                for i in item["positions"]:
                    vectors[i] = vector
        client = QdrantClient(path=str(root / "data/qdrant_figures_v2"))
        try:
            count = index_chunks(client, collection, figures, np.asarray(vectors), "CLIP-multilingual-512", signature)
        finally:
            client.close()
    else:
        count = 0
    result = {"collection": collection, "text_index_sha256": state_signature(text_state),
              "encoder_signature": signature, "dimensions": 512, "count": count, "figures": figures}
    destination = root / "data/index_figures_v2.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return result


if __name__ == "__main__":
    result = build_figure_index()
    print(f"{result['count']} régions figuratives indexées ; aucune interprétation de leur signification n'est ajoutée.")
