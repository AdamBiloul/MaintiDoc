import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from maintidoc.adaptation import compare_rankings, ranking_metrics, validate_dataset, verify_source
from maintidoc.reranker import LocalReranker, verify_base
from maintidoc.rag import OllamaGenerator
from maintidoc.settings import Settings
from scripts.prepare_adaptation import demo_dataset
import test_v1 as fixtures


class MetricTests(unittest.TestCase):
    def test_top_reranked_passage_cannot_starve_a_more_specific_fragment(self):
        import httpx
        response = httpx.Response(200, request=httpx.Request("POST", "http://localhost"),
                                  json={"message": {"content": '{"evidence_ids": []}'}})
        generic = " ".join(["Le bot affiche les informations disponibles."] * 8)
        specific = "Les commandes du bot sont /approve pour approuver et /cancel pour annuler."
        hits = [{"document": "d.pdf", "page": 1, "text": generic, "score": 0.9, "rerank_position": 0},
                {"document": "d.pdf", "page": 2, "text": specific, "score": 0.8, "rerank_position": 1}]
        with patch("maintidoc.rag.httpx.post", return_value=response) as post:
            OllamaGenerator(Settings())("Quelles commandes permettent d'approuver et d'annuler ?", hits)
        passages = json.loads(post.call_args.kwargs["json"]["messages"][1]["content"])["passages"]
        self.assertTrue(any(p["text"] == specific for p in passages))

    def test_graded_ndcg_missing_positive_and_cutoff(self):
        self.assertEqual(ranking_metrics(["a"], {"a": 1, "missing": 3})["mrr@10"], 1)
        self.assertLess(ranking_metrics(["a"], {"a": 1, "missing": 3})["ndcg@5"], 0.15)
        self.assertEqual(ranking_metrics(["b"], {"b": 0, "missing": 1}), {"ndcg@5": 0, "mrr@10": 0})
        gold = {str(i): int(i == 10) for i in range(11)}
        self.assertEqual(ranking_metrics(list(gold), gold), {"ndcg@5": 0, "mrr@10": 0})
        with self.assertRaises(ValueError):
            ranking_metrics(["a", "a"], {"a": 1})

    def test_three_modes_have_identical_candidates_without_injected_positives(self):
        query = {"id": "q", "query": "Question", "gold": {"a": 0, "b": 1, "missing": 3},
                 "candidates": [{"id": "a", "text": "negative", "retrieval_score": 1},
                                {"id": "b", "text": "positive", "retrieval_score": 0.5}]}
        calls = []
        def scorer(mode, question, passages):
            calls.append((mode, passages))
            return [0., 1.]
        report = compare_rankings([query], scorer)
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertFalse(report["positives_injected"])
        self.assertEqual(report["means"]["sans_reranker"]["mrr@10"], 0.5)
        self.assertEqual(report["means"]["initial"]["mrr@10"], 1.)
        for order in report["details"][0]["rankings"].values():
            self.assertEqual(set(order), {"a", "b"})


class DatasetTests(unittest.TestCase):
    def test_source_must_match_authorized_snapshot_and_actual_pdf_page(self):
        import hashlib
        import pymupdf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/raw").mkdir(parents=True)
            path = root / "data/raw/source.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((50, 50), "A synthetic source passage with verifiable provenance.")
                pdf.save(path)
            source = {"document_id": "d", "version": "v1", "path": "data/raw/source.pdf",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "page": 1}
            state = {"documents": [{"id": "d", "authorized": True, **{k: v for k, v in source.items() if k not in {"document_id", "page"}}}]}
            (root / "data/index_v1.json").write_text(json.dumps(state))
            verify_source(source, "A synthetic source passage", root, {})
            with self.assertRaises(ValueError):
                verify_source(source, "A fabricated passage absent from the PDF", root, {})
            with self.assertRaises(ValueError):
                verify_source(dict(source, version="v2"), "A synthetic source passage", root, {})
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_source(source, "A synthetic source passage", root, {})

    def test_changed_base_weights_are_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".models/reranker-base").mkdir(parents=True)
            (root / "data/evaluation").mkdir(parents=True)
            weights = root / ".models/reranker-base/model.safetensors"
            weights.write_bytes(b"fixture-weights")
            record = {"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
            (root / "data/evaluation/reranker_model.json").write_text(json.dumps(record))
            self.assertEqual(verify_base(root), record)
            weights.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_base(root)

    def test_demo_requires_explicit_flag_and_cannot_pass_business_validation(self):
        data = demo_dataset()
        self.assertEqual(validate_dataset(data, demo=True)["pairs"], 24)
        with self.assertRaises(ValueError):
            validate_dataset(data)
        data["demo_only"] = False
        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_leakage_duplicate_ids_and_invalid_labels_rejected(self):
        mutations = [
            lambda d: d["queries"][4].update(group=d["queries"][0]["group"]),
            lambda d: d["queries"][4].update(query=d["queries"][0]["query"]),
            lambda d: d["queries"][4]["candidates"][0].update(text=d["queries"][0]["candidates"][0]["text"]),
            lambda d: d["queries"][0]["gold"].update({"demo-0-0": None}),
            lambda d: d["queries"][0]["candidates"][0].update(retrieval_score=float("nan")),
            lambda d: d["queries"][1].update(id=d["queries"][0]["id"]),
        ]
        for mutate in mutations:
            data = copy.deepcopy(demo_dataset())
            mutate(data)
            with self.assertRaises(ValueError):
                validate_dataset(data, demo=True)

    def test_demo_checkpoint_is_refused_for_application(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / ".models/demo"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adaptation.json").write_text(json.dumps({"demo_only": True}))
            with self.assertRaises(ValueError):
                LocalReranker(root, "adapte", checkpoint)


class RerankingAccessTests(unittest.TestCase):
    generate = fixtures.AccessTests.generate

    def setUp(self):
        fixtures.AccessTests.setUp(self)

    def tearDown(self):
        fixtures.AccessTests.tearDown(self)

    def test_only_accessible_hits_reach_reranker_and_threshold_still_applies(self):
        reranker = LocalReranker(self.root)
        self.engine.reranker = reranker
        with patch.object(reranker, "scores", return_value=[-3.]) as scores:
            answer = self.engine.ask("Question de validation", self.identity)
            self.assertEqual(answer.status, "answered")
            self.assertEqual(scores.call_args.args[1], [self.chunks[0]["text"]])
            self.assertEqual(self.hits_seen[0][0]["score"], 0.8)
            self.assertEqual(self.hits_seen[0][0]["rerank_score"], -3.)
            self.settings.threshold = 0.9
            self.assertEqual(self.engine.ask("Question de validation", self.identity).status, "abstained")
            self.assertEqual(scores.call_count, 1)


if __name__ == "__main__":
    unittest.main()
