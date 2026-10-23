"""Séance 4 : indexer et rechercher les passages dans Qdrant local."""

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from qdrant_client import QdrantClient, models


def corpus_context(source, model):
    local = Path(".models/bge-m3")
    model_source = model
    if model == "BAAI/bge-m3" and any(
        (local / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        model_source = str(local.resolve())
    raw = source.read_bytes()
    chunks = json.loads(raw)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Le corpus doit contenir une liste non vide de passages.")
    key = hashlib.sha256(raw + model_source.encode()).hexdigest()
    # Une collection distincte par corpus et modèle évite de mélanger les versions.
    return chunks, model_source, key, f"maintidoc_{key}"


def index_chunks(client, collection, chunks, vectors, model, corpus_key):
    if vectors.ndim != 2 or len(vectors) != len(chunks) or vectors.shape[1] < 1:
        raise ValueError("Les vecteurs doivent correspondre aux passages, dans le même ordre.")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Les vecteurs doivent être finis et non nuls.")
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])) for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Le corpus contient des identifiants de passage en double.")
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        )
    config = client.get_collection(collection).config.params.vectors
    if config.size != vectors.shape[1] or config.distance != models.Distance.COSINE:
        raise ValueError("La collection existante utilise une configuration incompatible.")
    # Identifiants déterministes : relancer l'indexation ne crée pas de doublons.
    for start in range(0, len(chunks), 64):
        points = [
            models.PointStruct(
                id=ids[i], vector=vectors[i].tolist(),
                payload={**chunks[i], "embedding_model": model, "corpus_key": corpus_key},
            )
            for i in range(start, min(start + 64, len(chunks)))
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
    return client.count(collection, exact=True).count


def search(client, collection, vector, limit=3, document=None):
    query_filter = None
    if document is not None:
        query_filter = models.Filter(must=[models.FieldCondition(
            key="document", match=models.MatchValue(value=document),
        )])
    hits = client.query_points(
        collection_name=collection, query=vector.tolist(), limit=limit,
        query_filter=query_filter, with_payload=True,
    ).points
    return [dict(hit.payload, score=float(hit.score)) for hit in hits]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "search"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--source", type=Path, default=Path("data/chunks/feuille_de_route.json"))
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--db", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--document", help="Filtre documentaire de démonstration, sans contrôle d'identité")
    args = parser.parse_args()
    if args.top_k < 1 or (args.action == "search" and not (args.question or "").strip()):
        parser.error("La recherche nécessite une question et top-k doit être positif.")
    client = None
    try:
        chunks, model_source, key, collection = corpus_context(args.source, args.model)
        client = QdrantClient(path=str(args.db))
        if args.action == "index":
            cache = Path("data/embeddings") / f"{key}.npz"
            if not cache.is_file():
                raise ValueError("Cache absent : lancez d'abord semantic_search.py sur ce corpus et ce modèle.")
            with np.load(cache, allow_pickle=False) as saved:
                count = index_chunks(client, collection, chunks, saved["vectors"], args.model, key)
            print(f"{count} passages enregistrés dans {collection}.")
            print(f"Stockage persistant : {args.db}")
        else:
            if not client.collection_exists(collection):
                raise ValueError("Cette collection n'existe pas : lancez d'abord la commande index.")
            from sentence_transformers import SentenceTransformer
            print("Calcul du vecteur de la question avec BGE-M3…", flush=True)
            model = SentenceTransformer(model_source, device="cpu", cache_folder=".models")
            vector = model.encode([args.question], normalize_embeddings=True)[0]
            results = search(client, collection, vector, args.top_k, args.document)
            output = Path("data/search/qdrant_latest.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "question": args.question, "model": args.model,
                "collection": collection, "results": results,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            for rank, result in enumerate(results, 1):
                print(f"\n{rank}. {result['document']} — page {result['page']} "
                      f"(similarité : {result['score']:.3f})\n{result['text']}")
            if not results:
                print("Aucun passage trouvé avec ce filtre.")
            print(f"\nRésultat enregistré : {output}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Opération impossible : {error}\n")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
