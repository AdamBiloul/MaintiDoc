"""Vérifications du stockage local, sans télécharger de modèle."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from qdrant_client import QdrantClient

from scripts.qdrant_store import index_chunks, search


class QdrantStorageTests(unittest.TestCase):
    def test_persistence_idempotence_references_and_filter(self):
        chunks = [
            {"id": "a", "document": "pompe.pdf", "page": 7, "text": "Voyant orange"},
            {"id": "b", "document": "moteur.pdf", "page": 3, "text": "Inspection"},
        ]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "qdrant")
            client = QdrantClient(path=path)
            try:
                self.assertEqual(index_chunks(client, "test", chunks, vectors, "test-model", "v1"), 2)
                self.assertEqual(index_chunks(client, "test", chunks, vectors, "test-model", "v1"), 2)
            finally:
                client.close()
            # Réouverture : les données doivent avoir survécu à la fermeture.
            client = QdrantClient(path=path)
            try:
                hits = search(client, "test", np.array([1.0, 0.0]))
                self.assertEqual(hits[0]["id"], "a")
                self.assertEqual(hits[0]["page"], 7)
                self.assertEqual(hits[0]["text"], "Voyant orange")
                self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)
                filtered = search(client, "test", np.array([1.0, 0.0]), document="moteur.pdf")
                self.assertEqual([hit["id"] for hit in filtered], ["b"])
                self.assertEqual(search(client, "test", vectors[0], document="absent.pdf"), [])
                with self.assertRaises(ValueError):
                    index_chunks(client, "test", chunks, np.ones((2, 3)), "test-model", "v1")
                self.assertEqual(client.count("test", exact=True).count, 2)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
