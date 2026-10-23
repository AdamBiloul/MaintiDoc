"""Exercice 2 : découper les pages extraites en passages avec références."""

import argparse
import hashlib
import json
from pathlib import Path


def chunk_pages(pages: list[dict], size: int = 120, overlap: int = 25) -> list[dict]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Il faut size > 0 et 0 <= overlap < size.")
    chunks = []
    for page in pages:
        document, number, text = page["document"], page["page"], page["text"]
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Chaque page doit avoir un nom de document.")
        if type(number) is not int or number < 1 or not isinstance(text, str):
            raise ValueError("Chaque page doit avoir un numéro positif et un texte.")
        words = text.split()
        # Aucun passage ne traverse une page : la citation reste simple.
        for start in range(0, len(words), size - overlap):
            end = min(start + size, len(words))
            passage = " ".join(words[start:end])
            identity = json.dumps(
                [document, number, start, end, passage], ensure_ascii=False
            )
            chunks.append({
                "id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "document": document,
                "page": number,
                "word_start": start,
                "word_end": end,
                "text": passage,
            })
            if end == len(words):
                break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON produit par extract_pdf.py")
    parser.add_argument("--size", type=int, default=120, help="Mots par passage")
    parser.add_argument("--overlap", type=int, default=25, help="Mots communs entre passages")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("data/chunks") / args.source.name
    if output.resolve() == args.source.resolve():
        parser.error("La sortie doit être différente du fichier source.")
    try:
        pages = json.loads(args.source.read_text(encoding="utf-8"))
        if not isinstance(pages, list):
            raise ValueError("Le JSON doit contenir une liste de pages.")
        chunks = chunk_pages(pages, args.size, args.overlap)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Découpage impossible : {error}\n")
    print(f"{len(chunks)} passages créés à partir de {len(pages)} pages : {output}")


if __name__ == "__main__":
    main()
