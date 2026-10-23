"""Exercice 3 : recherche locale par embeddings, avant l'index Qdrant."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--source", type=Path, default=Path("data/chunks/feuille_de_route.json"))
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if not args.question.strip() or args.top_k < 1:
        parser.error("La question doit être renseignée et top-k doit être positif.")
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model_source = args.model
        local_model = Path(".models/bge-m3")
        if args.model == "BAAI/bge-m3" and any(
            (local_model / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            model_source = str(local_model.resolve())
        raw = args.source.read_bytes()
        chunks = json.loads(raw)
        if not chunks or not isinstance(chunks, list):
            raise ValueError("La source doit contenir une liste non vide de passages.")
        # Cache distinct pour chaque corpus et chaque modèle.
        key = hashlib.sha256(raw + model_source.encode()).hexdigest()
        cache = Path("data/embeddings") / f"{key}.npz"
        print(f"Chargement du modèle {args.model} sur CPU…", flush=True)
        model = SentenceTransformer(model_source, device="cpu", cache_folder=".models")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                vectors = saved["vectors"]
            print("Vecteurs des passages relus depuis le cache.", flush=True)
        else:
            vectors = model.encode(
                [c["text"] for c in chunks], batch_size=2,
                normalize_embeddings=True, show_progress_bar=True,
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, vectors=vectors)
        print(f"{vectors.shape[0]} passages, {vectors.shape[1]} nombres par vecteur.", flush=True)
        query = model.encode([args.question], normalize_embeddings=True)[0]
        # Le produit scalaire de vecteurs normalisés donne la similarité cosinus.
        scores = vectors @ query
        indices = np.argsort(-scores, kind="stable")[:args.top_k]
        results = [dict(chunks[int(i)], score=float(scores[i])) for i in indices]
        destination = Path("data/search/latest.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "question": args.question, "model": args.model,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. {result['document']} — page {result['page']} "
                  f"(similarité : {result['score']:.3f})")
            print(result["text"])
        print(f"\nRésultat enregistré : {destination}")
    except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Recherche impossible : {error}\n")


if __name__ == "__main__":
    main()
