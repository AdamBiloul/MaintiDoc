"""Recette HTTP de la fusion, des figures et de leurs références dans le corpus actif."""
import base64
from io import BytesIO
import json
from pathlib import Path
import time

import httpx
from PIL import Image

root = Path(__file__).resolve().parents[1]
base = "http://127.0.0.1:8000"
headers = {"Authorization": "Bearer " + (root / "data/runtime/demo-token.txt").read_text(encoding="utf-8").strip()}
with httpx.Client(timeout=600) as client:
    assert client.post(f"{base}/search-multimodal", json={}).status_code == 401
    documents = client.get(f"{base}/documents", headers=headers)
    documents.raise_for_status()
    allowed = {d["id"] for d in documents.json()}
    started = time.perf_counter()
    response = client.post(f"{base}/search-multimodal", headers=headers,
                           json={"question": "planning des étapes du projet", "top_k": 8})
    response.raise_for_status()
    first = response.json()
    text_seconds = round(time.perf_counter() - started, 2)
    assert first["figures_index_status"] == "ready"
    assert first["pages"] and all(p["document_id"] in allowed for p in first["pages"])
    matches = [(p, f) for p in first["pages"] for f in p["figures"]]
    assert matches
    page, figure = matches[0]
    asset = client.get(f"{base}/figures/{figure['figure_id']}/image", headers=headers)
    asset.raise_for_status()
    with Image.open(BytesIO(asset.content)) as image:
        assert image.format == "PNG" and max(image.size) <= 1024
    assert client.get(f"{base}/figures/{'0' * 64}/image", headers=headers).status_code == 404
    started = time.perf_counter()
    response = client.post(f"{base}/search-multimodal", headers=headers, json={
        "question": "retrouver cette figure", "image_base64": base64.b64encode(asset.content).decode("ascii"), "top_k": 8,
    })
    response.raise_for_status()
    second = response.json()
    image_seconds = round(time.perf_counter() - started, 2)
    repeated = [f for p in second["pages"] for f in p["figures"] if f["figure_id"] == figure["figure_id"]]
    assert repeated and repeated[0]["scores"]["figure_image"] > 0.99
    pdf = client.get(f"{base}/documents/{page['document_id']}/pdf", headers=headers)
    pdf.raise_for_status()
report = {"all_passed": True, "corpus": "Corpus actif, feuille de route lors de cette recette",
          "authentication_required": True, "unknown_asset_rejected": True,
          "asset_downloaded": True, "pdf_downloaded": True,
          "text_search_seconds": text_seconds, "text_image_search_seconds": image_seconds,
          "text_result": first, "image_result": second,
          "limitation": "L'image de test est un extrait identique de l'index ; aucune précision métier ou résistance aux changements de perspective n'est établie."}
(root / "data/evaluation/v2_multimodal_live.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Fusion et références vérifiées : texte {text_seconds} s ; texte + image {image_seconds} s.")
