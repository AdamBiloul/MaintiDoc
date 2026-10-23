"""Recette du premier incrément photo, avec fixture géométrique synthétique."""
import base64
import hashlib
import json
from pathlib import Path
import time

import httpx
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
fixture = root / "data/evaluation/vision_fixture.png"
fixture.parent.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (384, 256), "#eeeeee")
draw = ImageDraw.Draw(image)
draw.rectangle((30, 30, 354, 226), fill="#333333", outline="black", width=3)
draw.ellipse((80, 90, 140, 150), fill="#f09519")
draw.rectangle((220, 90, 290, 150), fill="#777777")
image.save(fixture)
raw = fixture.read_bytes()
token = (root / "data/runtime/demo-token.txt").read_text(encoding="utf-8").strip()
base = "http://127.0.0.1:8000"
headers = {"Authorization": f"Bearer {token}"}
started = time.perf_counter()
with httpx.Client(timeout=240) as client:
    assert client.post(f"{base}/analyze-image", json={}).status_code == 401
    invalid = client.post(f"{base}/analyze-image", headers=headers,
                          json={"question": "Quel voyant ?", "image_base64": "invalid%%"})
    assert invalid.status_code == 400
    response = client.post(f"{base}/analyze-image", headers=headers, json={
        "question": "Que signifie ce voyant orange ?",
        "image_base64": base64.b64encode(raw).decode("ascii"),
    })
    response.raise_for_status()
    result = response.json()
assert result["status"] == "clarification_required"
assert result["observation_verified"] is False
assert result["observation"]
assert result["image_sha256"] == hashlib.sha256(raw).hexdigest()
assert result["documentary_answer"]["citations"] == []
report = {"fixture": "Dessin synthétique, pas une photographie métier",
          "seconds": round(time.perf_counter() - started, 2), "all_passed": True,
          "authentication_checked": True, "invalid_image_rejected": True,
          "semantic_accuracy_on_maintenance_photos_validated": False, "response": result}
(root / "data/evaluation/v2_photo_live.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Recette photo réussie : VLM exécuté, clarification demandée, aucune citation inventée ; {report['seconds']} s.")
