"""Premier incrément V2 : observations non vérifiées et réponse documentaire contrôlée."""
import base64
import binascii
import hashlib
from io import BytesIO
import json
import os
import subprocess
import sys

from PIL import Image, ImageOps, UnidentifiedImageError

from maintidoc.rag import abstain
from maintidoc.schemas import ImageAnswer

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_BODY_BYTES = 8 * 1024 * 1024


class InvalidImage(ValueError):
    pass


def clean_image(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidImage("L'image doit être encodée en base64 valide.")
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImage("Image vide ou supérieure à 5 Mo.")
    try:
        with Image.open(BytesIO(raw)) as candidate:
            if candidate.format not in {"JPEG", "PNG"}:
                raise InvalidImage("Formats acceptés : JPEG et PNG.")
            if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
                raise InvalidImage("Image supérieure à 12 millions de pixels.")
            candidate.verify()
        with Image.open(BytesIO(raw)) as candidate:
            oriented = ImageOps.exif_transpose(candidate).convert("RGB")
            oriented.thumbnail((1024, 1024))
            clean = Image.new("RGB", oriented.size)
            clean.paste(oriented)
            output = BytesIO()
            clean.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise InvalidImage("Image illisible ou corrompue.")
    return output.getvalue(), hashlib.sha256(raw).hexdigest()


class LocalVision:
    name = "HuggingFaceTB/SmolVLM-256M-Instruct"

    def __init__(self, settings):
        self.settings = settings

    def __call__(self, sanitized):
        model_path = self.settings.root / ".models/smolvlm"
        if not (model_path / "model.safetensors").is_file():
            raise RuntimeError("Le modèle de vision n'est pas installé.")
        try:
            process = subprocess.run(
                [sys.executable, str(self.settings.root / "scripts/vision_worker.py"), str(model_path)],
                input=json.dumps({"image": base64.b64encode(sanitized).decode("ascii")}),
                capture_output=True, text=True, encoding="utf-8", timeout=180,
                cwd=self.settings.root,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                         PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Le calcul visuel a dépassé le délai autorisé.")
        if process.returncode:
            raise RuntimeError("Le modèle de vision local n'a pas pu analyser l'image.")
        try:
            observation = json.loads(process.stdout)["observation"].strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            raise RuntimeError("Sortie du modèle de vision invalide.")
        if not observation or len(observation) > 2000:
            raise RuntimeError("Observation visuelle invalide.")
        return observation


class PhotoAssistant:
    def __init__(self, engine, settings, vision=None):
        self.engine = engine
        self.vision = vision or LocalVision(settings)

    def analyze(self, request, identity):
        sanitized, digest = clean_image(request.image_base64)
        # Même verrou que le texte : VLM, embeddings et génération restent séquentiels.
        with self.engine._lock:
            observation = self.vision(sanitized)
            reference = (request.equipment_reference.strip() or None) if request.equipment_reference else None
            documents = self.engine.documents(identity)
            matching = [d["id"] for d in documents if reference and
                        reference.casefold() in {str(r).casefold() for r in d.get("equipment_ids", [])}]
            if not reference:
                answer = abstain("equipement_non_identifie")
                clarification = "Quelle est la référence exacte de l'équipement photographié ? Ajoutez aussi son manuel autorisé."
            elif not matching:
                answer = abstain("manuel_equipement_absent")
                clarification = "Aucun manuel accessible n'est associé à cette référence. Ajoutez le manuel et son association à l'équipement."
            else:
                query = f"{request.question}\nRéférence fournie : {reference}\nObservation automatique non vérifiée : {observation}"
                answer = self.engine.chain.invoke({"question": query, "identity": identity,
                                                   "document_ids": matching, "visual_query": sanitized})
                clarification = None if answer.status == "answered" else "Précisez le repère du voyant ou fournissez un gros plan lisible et la légende du panneau."
            return ImageAnswer(
                status="clarification_required" if clarification else answer.status,
                observation=observation, model=LocalVision.name, image_sha256=digest,
                equipment_reference=reference, clarification=clarification,
                uncertainties=["Observation automatique en anglais, non vérifiée.",
                               "La photographie ne prouve pas la référence de l'équipement ni la signification d'un voyant.",
                               "Les couleurs, symboles et inscriptions peuvent être mal interprétés par ce petit modèle."],
                documentary_answer=answer,
            )
