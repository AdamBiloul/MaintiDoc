"""Interface Streamlit : rendu et erreurs avec transport API simulé."""
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from io import BytesIO
from PIL import Image
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "ui/streamlit_app.py"


def response(status, json=None, content=None):
    return httpx.Response(status, json=json, content=content,
                          request=httpx.Request("GET", "http://127.0.0.1:8000"))


class InterfaceTests(unittest.TestCase):
    def test_figure_explorer_renders_reference_and_authorized_preview(self):
        png = BytesIO()
        Image.new("RGB", (20, 20), "orange").save(png, format="PNG")
        result = {"figures_index_status": "ready", "notice": "Correspondances à vérifier.", "pages": [{
            "document_id": "roadmap", "document": "fixture.pdf", "version": "test", "page": 2,
            "fusion_score": 0.03, "modalities": ["text", "figure_text"],
            "text_excerpts": ["Extrait documentaire de test."],
            "figures": [{"figure_id": "a" * 64, "kind": "raster", "bbox": [0, 0, 20, 20], "scores": {}}],
        }]}
        def get(url, **kwargs):
            if url.endswith("/documents"):
                return response(200, json=[{"id": "roadmap"}])
            return response(200, content=png.getvalue() if "/figures/" in url else b"%PDF fixture")
        with patch("httpx.get", side_effect=get), patch("httpx.post", return_value=response(200, json=result)) as post:
            app = AppTest.from_file(str(APP)).run(timeout=30)
            app.text_area[0].input("Rechercher une figure")
            app.checkbox[0].check()
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.expander), 1)
            self.assertIn("Extrait documentaire", " ".join(item.value for item in app.text))
            self.assertTrue(post.call_args.args[0].endswith("/search-multimodal"))

    def test_question_answer_citation_and_pdf_render(self):
        answer = {"status": "answered", "answer": "Une validation humaine est obligatoire. [S1]",
                  "notice": "Démonstration", "citations": [{"label": "S1", "document_id": "roadmap",
                  "document": "fixture.pdf", "version": "test", "page": 3,
                  "quote": "Une validation humaine est obligatoire.", "score": 0.8}]}

        def get(url, **kwargs):
            if url.endswith("/documents"):
                return response(200, json=[{"id": "roadmap"}])
            return response(200, content=b"%PDF-1.7 fixture")

        with patch("httpx.get", side_effect=get), patch("httpx.post", return_value=response(200, json=answer)):
            app = AppTest.from_file(str(APP)).run(timeout=30)
            app.text_area[0].input("Une validation est-elle obligatoire ?")
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.expander), 1)
            self.assertIn("validation humaine", " ".join(item.value for item in app.text))

    def test_denied_session_shows_error_without_question_form(self):
        with patch("httpx.get", return_value=response(401, json={"detail": "Unauthorized"})):
            app = AppTest.from_file(str(APP)).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.text_area), 0)

    def test_photo_observation_and_clarification_are_distinct_from_document_answer(self):
        with patch("httpx.get", return_value=response(200, json=[{"id": "roadmap"}])):
            app = AppTest.from_file(str(APP)).run(timeout=30)
            app.session_state["answer_token"] = app.sidebar.text_input[0].value
            app.session_state["answer"] = {"status": "abstained", "answer": "Pas de preuve documentaire.", "notice": "Démonstration", "citations": []}
            app.session_state["photo"] = {"observation": "An orange circle on a panel.",
                                          "clarification": "Indiquez la référence exacte de l'équipement.",
                                          "uncertainties": ["Observation non vérifiée."]}
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.info), 1)
            self.assertEqual(len(app.warning), 0)
            self.assertIn("orange circle", " ".join(item.value for item in app.text))


if __name__ == "__main__":
    unittest.main()
