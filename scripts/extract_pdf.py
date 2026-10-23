"""Exercice 1 : extraire un PDF en conservant ses pages."""

import argparse
import json
from pathlib import Path


def extract_pdf(source: Path, destination: Path) -> None:
    import pymupdf

    # Chaque page garde son numéro et le nom du document pour les citations.
    with pymupdf.open(source) as document:
        if document.needs_pass:
            raise ValueError("Le PDF est protégé par un mot de passe.")
        pages = [
            {
                "document": source.name,
                "page": number,
                "text": page.get_text().strip(),
            }
            for number, page in enumerate(document, start=1)
        ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    empty_pages = [page["page"] for page in pages if not page["text"]]
    print(f"{len(pages)} pages enregistrées dans {destination}")
    if empty_pages:
        print(f"Pages sans texte extrait : {empty_pages}")
        print("Ces pages peuvent nécessiter un OCR ou contenir seulement des images.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Chemin du PDF à lire")
    parser.add_argument("--output", type=Path, help="Chemin du fichier JSON produit")
    args = parser.parse_args()
    if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
        parser.error("Indiquez le chemin d'un fichier PDF existant.")
    output = args.output or Path("data/extracted") / f"{args.pdf.stem}.json"
    if output.resolve() == args.pdf.resolve():
        parser.error("Le fichier de sortie doit être différent du PDF source.")
    try:
        extract_pdf(args.pdf, output)
    except ImportError:
        parser.exit(1, "Installez PyMuPDF avec : python -m pip install -r requirements.txt\n")
    except Exception as error:
        parser.exit(1, f"Extraction impossible : {error}\n")


if __name__ == "__main__":
    main()
