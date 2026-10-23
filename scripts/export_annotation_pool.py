"""Exporte les passages sources pour annotation, sans attribuer de pertinence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chunk_text import chunk_pages
import pymupdf

state = json.loads((ROOT / "data/index_v1.json").read_text(encoding="utf-8"))
records = []
for document in state["documents"]:
    path = (ROOT / document["path"]).resolve()
    if not document.get("authorized") or not path.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Document non autorisé.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != document["sha256"]:
        raise ValueError("PDF changé : réindexation requise.")
    with pymupdf.open(path) as pdf:
        pages = [{"document": path.name, "page": i, "text": p.get_text()} for i, p in enumerate(pdf, 1)]
    for chunk in chunk_pages(pages):
        records.append({"id": hashlib.sha256(f"{document['id']}:{document['version']}:{chunk['id']}".encode()).hexdigest(),
                        "text": chunk["text"], "source": {"document_id": document["id"],
                            "version": document["version"], "path": document["path"],
                            "sha256": document["sha256"], "page": chunk["page"]},
                        "allowed_roles": document["allowed_roles"], "relevance": None,
                        "notice": "Passage non annoté ; ne pas confondre avec un exemple vérifié."})
destination = ROOT / "data/adaptation" / f"passage_pool_{state['collection']}.json"
destination.parent.mkdir(exist_ok=True)
if destination.exists():
    raise SystemExit("Export déjà présent ; aucune annotation écrasée.")
destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(records)} passages non annotés exportés dans {destination.relative_to(ROOT)}.")
