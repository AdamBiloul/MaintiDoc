import base64
import hashlib
from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

import test_v1 as fixtures
from maintidoc.api import create_app
from maintidoc.schemas import ImageQuestion
from maintidoc.vision import clean_image, InvalidImage, PhotoAssistant, MAX_BODY_BYTES


def image_data(format="PNG"):
    image = Image.new("RGB", (40, 20), "orange")
    buffer = BytesIO()
    exif = Image.Exif()
    exif[271] = "Private fixture camera metadata"
    image.save(buffer, format=format, exif=exif)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ImageValidationTests(unittest.TestCase):
    def test_valid_image_is_reencoded_without_metadata(self):
        encoded = image_data("JPEG")
        clean, digest = clean_image(encoded)
        self.assertEqual(digest, hashlib.sha256(base64.b64decode(encoded)).hexdigest())
        with Image.open(BytesIO(clean)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGB")
            self.assertFalse(image.getexif())

    def test_invalid_base64_corrupt_unsupported_and_too_large_images(self):
        for value in ("invalid%%%", base64.b64encode(b"not an image").decode()):
            with self.assertRaises(InvalidImage):
                clean_image(value)
        with self.assertRaises(InvalidImage):
            clean_image(image_data("GIF"))
        with patch("maintidoc.vision.MAX_IMAGE_BYTES", 1):
            with self.assertRaises(InvalidImage):
                clean_image(image_data())
        with patch("maintidoc.vision.MAX_IMAGE_PIXELS", 1):
            with self.assertRaises(InvalidImage):
                clean_image(image_data())


class PhotoTests(unittest.TestCase):
    generate = fixtures.AccessTests.generate

    def setUp(self):
        fixtures.AccessTests.setUp(self)
        self.vision_calls = []
        def vision(image):
            self.vision_calls.append(image)
            return "An orange circular object on a panel."
        self.photo = PhotoAssistant(self.engine, self.settings, vision=vision)

    def tearDown(self):
        fixtures.AccessTests.tearDown(self)

    def test_unknown_equipment_requires_clarification_without_document_generation(self):
        result = self.photo.analyze(ImageQuestion(question="Quel voyant ?", image_base64=image_data()), self.identity)
        self.assertEqual(result.status, "clarification_required")
        self.assertFalse(result.observation_verified)
        self.assertEqual(result.documentary_answer.citations, [])
        self.assertEqual(self.hits_seen, [])
        self.assertEqual(len(self.vision_calls), 1)

    def test_inaccessible_equipment_manual_is_not_disclosed(self):
        self.engine.state["documents"][1]["equipment_ids"] = ["PRIVATE-MOTOR"]
        result = self.photo.analyze(ImageQuestion(question="Quel voyant ?", image_base64=image_data(), equipment_reference="PRIVATE-MOTOR"), self.identity)
        self.assertEqual(result.documentary_answer.reason, "manuel_equipement_absent")
        self.assertNotIn("SECRET_FIXTURE", result.model_dump_json())
        self.assertEqual(self.hits_seen, [])

    def test_matching_equipment_keeps_acl_filter_before_generation(self):
        self.engine.state["documents"][0]["equipment_ids"] = ["PUMP-A"]
        result = self.photo.analyze(ImageQuestion(question="Quelle validation ?", image_base64=image_data(), equipment_reference="pump-a"), self.identity)
        self.assertEqual(result.status, "answered")
        self.assertEqual([h["document_id"] for h in self.hits_seen[0]], ["public"])

    def test_endpoint_auth_validation_body_limit_and_clarification(self):
        with TestClient(create_app(self.settings, self.engine, photo=self.photo)) as client:
            self.assertEqual(client.post("/analyze-image", json={}).status_code, 401)
            headers = {"Authorization": "Bearer test-token"}
            self.assertEqual(client.post("/analyze-image", headers=headers, json={"question": "Quel voyant ?", "image_base64": "invalid%%"}).status_code, 400)
            self.assertEqual(client.post("/analyze-image", headers=headers, json={"question": "Quel voyant ?", "image_base64": image_data(), "roles": ["trainer"]}).status_code, 422)
            self.assertEqual(client.post("/analyze-image", headers=headers, content=b"x" * (MAX_BODY_BYTES + 1)).status_code, 413)
            response = client.post("/analyze-image", headers=headers, json={"question": "Quel voyant ?", "image_base64": image_data()})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "clarification_required")


if __name__ == "__main__":
    unittest.main()
