import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pymupdf
from PIL import Image
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from maintidoc.api import create_app
from maintidoc.figures import build_figure_index, extract_figures
from maintidoc.multimodal import MultimodalExplorer, StaleFigureIndex, fuse_rankings
from maintidoc.rag import RagEngine
from maintidoc.schemas import Draft, Identity, MultimodalQuestion
from maintidoc.settings import Settings
from scripts.qdrant_store import index_chunks


class FakeClip:
    signature = "fixture-clip-v1"

    def __init__(self):
        self.calls = 0

    def images(self, images):
        self.calls += 1
        vectors = np.zeros((len(images), 512), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors

    def text(self, texts):
        return self.images(texts)


class FakeText:
    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts])


class FusionTests(unittest.TestCase):
    def test_rrf_deduplicates_pages_and_keeps_versions_separate(self):
        hit = {"document_id": "d", "document": "d.pdf", "version": "v1", "page": 1, "text": "Evidence"}
        figure = {**hit, "id": "f", "bbox": [0., 0., 10., 10.], "kind": "raster", "score": 0.2}
        result = fuse_rankings({"text": [hit, hit, dict(hit, page=2)], "figure_text": [figure]})
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].fusion_score, 2 / 61)
        self.assertAlmostEqual(result[1].fusion_score, 1 / 62)
        self.assertEqual(result[0].modalities, ["text", "figure_text"])
        self.assertEqual(result[0].figures[0].figure_id, "f")
        versions = fuse_rankings({"text": [hit, dict(hit, version="v2")]})
        self.assertEqual(len(versions), 2)


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "data/raw").mkdir(parents=True)
        (self.root / "config").mkdir()
        image = Image.new("RGB", (64, 64), "orange")
        png = BytesIO()
        image.save(png, format="PNG")
        documents, chunks = [], []
        for identifier, role in [("public", "technician"), ("private", "trainer")]:
            path = self.root / f"data/raw/{identifier}.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_image(pymupdf.Rect(40, 50, 180, 190), stream=png.getvalue())
            page.draw_rect(pymupdf.Rect(240, 60, 360, 180), color=(0, 0, 1), width=2)
            page.insert_text((40, 240), f"Synthetic {identifier} procedure and panel legend.")
            pdf.save(path)
            pdf.close()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            documents.append({"id": identifier, "title": identifier, "path": f"data/raw/{identifier}.pdf",
                              "version": "v1", "sha256": digest, "allowed_roles": [role],
                              "pages": 1, "family": "synthetic", "equipment_ids": [identifier.upper()]})
            chunks.append({"id": identifier, "document_id": identifier, "document": path.name, "version": "v1",
                           "page": 1, "allowed_roles": [role], "content_sha256": digest,
                           "text": f"Synthetic {identifier} procedure and panel legend."})
        self.state = {"collection": "fixture", "documents": documents}
        (self.root / "data/index_v1.json").write_text(json.dumps(self.state), encoding="utf-8")
        (self.root / "config/users.local.json").write_text(json.dumps({"users": [{
            "user_id": "tester", "roles": ["technician"], "enabled": True,
            "token_sha256": hashlib.sha256(b"test-token").hexdigest(),
        }]}), encoding="utf-8")
        client = QdrantClient(path=str(self.root / "data/qdrant_v1"))
        index_chunks(client, "fixture", chunks, np.array([[0.8, 0.6], [1., 0.]]), "fake", "fixture")
        client.close()
        self.encoder = FakeClip()
        self.index = build_figure_index(self.root, self.encoder)
        self.settings = Settings(root=self.root, reranker_mode="off")
        self.engine = RagEngine(self.settings, FakeText(), lambda q, h: Draft(supported=False, claims=[]))
        self.explorer = MultimodalExplorer(self.engine, self.settings, self.encoder)
        self.identity = Identity(user_id="tester", roles=["technician"])

    def tearDown(self):
        self.explorer.close()
        self.engine.close()
        self.directory.cleanup()

    def test_extraction_metadata_pixels_cache_and_idempotence(self):
        self.assertEqual({f["kind"] for f in self.index["figures"]}, {"raster", "vector_region"})
        for figure in self.index["figures"]:
            self.assertEqual(figure["page"], 1)
            self.assertEqual(figure["version"], "v1")
            self.assertEqual(len(figure["bbox"]), 4)
            path = self.root / figure["image_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), figure["image_sha256"])
            with Image.open(path) as image:
                self.assertLessEqual(max(image.size), 1024)
        calls = self.encoder.calls
        second = build_figure_index(self.root, self.encoder)
        self.assertEqual(second["collection"], self.index["collection"])
        self.assertEqual(second["count"], self.index["count"])
        self.assertEqual(calls, self.encoder.calls)

    def test_photo_fusion_supplies_only_accessible_text_to_reranker(self):
        from unittest.mock import Mock
        self.engine.figure_store = self.explorer.figures
        reranker = Mock()
        reranker.rerank.side_effect = lambda question, hits: hits
        self.engine.reranker = reranker
        result = self.engine._retrieve({"question": "panel legend", "identity": self.identity,
                                        "document_ids": ["public"], "visual_query": b"fake-clip-fixture"})
        self.assertEqual({h["document_id"] for h in result["hits"]}, {"public"})
        self.assertEqual(reranker.rerank.call_count, 1)
        self.assertEqual({h["document_id"] for h in reranker.rerank.call_args.args[1]}, {"public"})
        self.assertGreater(self.encoder.calls, 1)

    def test_search_and_assets_keep_role_and_equipment_filters(self):
        result = self.explorer.search(MultimodalQuestion(question="panel legend"), self.identity)
        self.assertEqual({p.document_id for p in result.pages}, {"public"})
        self.assertTrue(any(p.figures for p in result.pages))
        private = next(f for f in self.index["figures"] if f["document_id"] == "private")
        with self.assertRaises(FileNotFoundError):
            self.explorer.figures.asset_path(private["id"], self.identity)
        self.assertEqual(self.explorer.search(MultimodalQuestion(question="panel", equipment_reference="PRIVATE"), self.identity).pages, [])
        calls = self.encoder.calls
        result = self.explorer.search(MultimodalQuestion(question="panel"), Identity(user_id="outsider", roles=["outsider"]))
        self.assertEqual(result.pages, [])
        self.assertEqual(self.encoder.calls, calls)

    def test_snapshot_and_model_changes_reject_stale_index(self):
        self.encoder.signature = "different-encoder"
        with self.assertRaises(StaleFigureIndex):
            self.explorer.search(MultimodalQuestion(question="panel"), self.identity)
        self.encoder.signature = "fixture-clip-v1"
        self.engine.state["documents"][0]["version"] = "v2"
        with self.assertRaises(StaleFigureIndex):
            self.explorer.search(MultimodalQuestion(question="panel"), self.identity)

    def test_asset_tampering_and_pdf_changes_are_detected(self):
        public = next(f for f in self.index["figures"] if f["document_id"] == "public")
        path = self.explorer.figures.asset_path(public["id"], self.identity)
        path.write_bytes(b"changed")
        with self.assertRaises(StaleFigureIndex):
            self.explorer.figures.asset_path(public["id"], self.identity)
        (self.root / "data/raw/public.pdf").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            extract_figures(self.root, self.state)

    def test_api_search_media_auth_and_validation(self):
        with TestClient(create_app(self.settings, self.engine, explorer=self.explorer)) as client:
            self.assertEqual(client.post("/search-multimodal", json={}).status_code, 401)
            headers = {"Authorization": "Bearer test-token"}
            self.assertEqual(client.post("/search-multimodal", headers=headers, json={"question": "panel", "roles": ["trainer"]}).status_code, 422)
            result = client.post("/search-multimodal", headers=headers, json={"question": "panel legend"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual({p["document_id"] for p in result.json()["pages"]}, {"public"})
            public = next(f for f in self.index["figures"] if f["document_id"] == "public")
            private = next(f for f in self.index["figures"] if f["document_id"] == "private")
            self.assertEqual(client.get(f"/figures/{public['id']}/image", headers=headers).status_code, 200)
            self.assertEqual(client.get(f"/figures/{private['id']}/image", headers=headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()
