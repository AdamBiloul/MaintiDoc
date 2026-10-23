"""Ingestion administrative : PDF autorisés, versions, droits et index Qdrant."""
import argparse
import hashlib
import json
from pathlib import Path

import pymupdf
from qdrant_client import QdrantClient

from maintidoc.embeddings import BGE
from maintidoc.settings import ROOT
from scripts.chunk_text import chunk_pages
from scripts.qdrant_store import index_chunks


def ingest(root=ROOT, manifest=None, ocr=False):
    root = Path(root)
    manifest = Path(manifest) if manifest else root / "config/corpus.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))["documents"]
    chunks, documents, empty_pages, identifiers = [], [], [], set()
    for entry in entries:
        if entry.get("authorized") is not True or not entry.get("allowed_roles") or not entry.get("version"):
            raise ValueError("Chaque document nécessite une autorisation, une version et des rôles.")
        identifier = entry["id"]
        if identifier in identifiers or not identifier.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Identifiant documentaire invalide ou en double.")
        identifiers.add(identifier)
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to((root / "data/raw").resolve()) or path.suffix.lower() != ".pdf":
            raise ValueError("Les documents doivent être des PDF placés dans data/raw.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pages = []
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"PDF protégé : {identifier}")
            for number, page in enumerate(pdf, 1):
                text = page.get_text().strip()
                if not text and ocr:
                    text = page.get_text(textpage=page.get_textpage_ocr(language="fra+eng")).strip()
                if not text:
                    empty_pages.append({"document_id": identifier, "page": number})
                pages.append({"document": path.name, "page": number, "text": text})
        extracted = root / "data/extracted_v1" / f"{identifier}.json"
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        passages = chunk_pages(pages)
        if not passages:
            raise ValueError(f"Aucun texte exploitable : {identifier}. Essayez --ocr avec Tesseract installé.")
        for passage in passages:
            passage.update(document_id=identifier, version=entry["version"],
                           allowed_roles=entry["allowed_roles"], content_sha256=digest,
                           family=entry.get("family", "non-renseignee"))
        chunks.extend(passages)
        documents.append({**entry, "sha256": digest, "pages": len(pages)})
    if not chunks:
        raise ValueError("Le manifeste ne contient aucun passage exploitable.")
    key = hashlib.sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    collection = f"v1_{key}"
    embedder = BGE(root)
    vectors = embedder.corpus_vectors(chunks)
    # Les identifiants de points intègrent l'identité documentaire et la version.
    indexed = [{**c, "id": hashlib.sha256(f"{c['document_id']}:{c['version']}:{c['id']}".encode()).hexdigest()} for c in chunks]
    client = QdrantClient(path=str(root / "data/qdrant_v1"))
    try:
        count = index_chunks(client, collection, indexed, vectors, BGE.name, key)
    finally:
        client.close()
    state = {"collection": collection, "model": BGE.name, "dimensions": int(vectors.shape[1]),
             "documents": documents, "points": count, "empty_pages": empty_pages}
    # La nouvelle collection n'est activée qu'une fois l'indexation terminée.
    destination = root / "data/index_v1.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ocr", action="store_true", help="OCR local des pages sans texte (Tesseract requis)")
    args = parser.parse_args()
    state = ingest(manifest=args.manifest, ocr=args.ocr)
    print(f"{len(state['documents'])} documents, {state['points']} passages, {len(state['empty_pages'])} pages sans texte.")
