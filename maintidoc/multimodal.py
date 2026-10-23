"""Recherche protégée texte/figure/photo et fusion par rangs RRF."""
import hashlib
import json

from qdrant_client import QdrantClient, models

from maintidoc.clip import ClipEncoder
from maintidoc.figures import state_signature
from maintidoc.schemas import FigureMatch, MatchedPage, MultimodalResult
from maintidoc.vision import clean_image


class StaleFigureIndex(ValueError):
    pass


def fuse_rankings(rankings, limit=5, k=60):
    """RRF sur pages uniques par liste ; aucune addition de scores cosinus bruts."""
    if k < 1 or limit < 1:
        raise ValueError("Les paramètres de fusion doivent être positifs.")
    pages = {}
    for modality, hits in rankings.items():
        seen, rank = set(), 0
        for hit in hits:
            key = (hit["document_id"], hit["version"], hit["page"])
            if key not in pages:
                pages[key] = {"document_id": hit["document_id"], "document": hit["document"],
                              "version": hit["version"], "page": hit["page"],
                              "fusion_score": 0.0, "modalities": [], "text_excerpts": [], "figures": {}}
            page = pages[key]
            if key not in seen:
                rank += 1
                page["fusion_score"] += 1 / (k + rank)
                page["modalities"].append(modality)
                seen.add(key)
            if modality == "text":
                excerpt = hit["text"][:800]
                if excerpt not in page["text_excerpts"] and len(page["text_excerpts"]) < 8:
                    page["text_excerpts"].append(excerpt)
            else:
                figure = page["figures"].setdefault(hit["id"], {
                    "figure_id": hit["id"], "kind": hit["kind"], "bbox": hit["bbox"], "scores": {},
                })
                figure["scores"][modality] = float(hit["score"])
    ordered = sorted(pages.values(), key=lambda p: (-p["fusion_score"], p["document_id"], p["page"]))
    return [MatchedPage(**{**p, "figures": [FigureMatch(**f) for f in p["figures"].values()]})
            for p in ordered[:limit]]


class FigureStore:
    def __init__(self, engine, settings, encoder=None):
        self.engine = engine
        self.settings = settings
        self.encoder = encoder or ClipEncoder(settings.root)
        path = settings.root / "data/index_figures_v2.json"
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        self.lookup = {f["id"]: f for f in self.state["figures"]} if self.state else {}
        self.client = None

    def close(self):
        if self.client is not None:
            self.client.close()

    @property
    def status(self):
        return "not_indexed" if self.state is None else ("ready" if self.state["count"] else "empty")

    def _check_snapshot(self):
        if self.state and self.state["text_index_sha256"] != state_signature(self.engine.state):
            raise StaleFigureIndex("Le corpus textuel a changé : réindexez les figures et redémarrez l'API.")

    def search(self, question, image, identity, document_ids, limit=8):
        self._check_snapshot()
        if self.status != "ready" or not document_ids:
            return {"figure_text": [], "figure_image": []}
        if self.state["encoder_signature"] != self.encoder.signature:
            raise StaleFigureIndex("Les encodeurs CLIP ont changé : réindexez les figures.")
        if self.client is None:
            self.client = QdrantClient(path=str(self.settings.root / "data/qdrant_figures_v2"))
        filters = models.Filter(must=[
            models.FieldCondition(key="allowed_roles", match=models.MatchAny(any=identity.roles)),
            models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)),
        ])
        # Ne charge pas de modèle si le rôle n'a accès à aucune figure.
        if not self.client.count(self.state["collection"], count_filter=filters, exact=True).count:
            return {"figure_text": [], "figure_image": []}
        vectors = {"figure_text": self.encoder.text([question])[0]}
        if image is not None:
            vectors["figure_image"] = self.encoder.images([image])[0]
        rankings = {}
        for name, vector in vectors.items():
            hits = self.client.query_points(self.state["collection"], query=vector.tolist(),
                                            query_filter=filters, limit=limit, with_payload=True).points
            rankings[name] = [dict(h.payload, score=float(h.score)) for h in hits]
        return rankings

    def asset_path(self, figure_id, identity):
        self._check_snapshot()
        figure = self.lookup.get(figure_id)
        allowed = {d["id"]: d for d in self.engine.documents(identity)}
        if (figure is None or figure["document_id"] not in allowed or
                not set(identity.roles).intersection(figure["allowed_roles"])):
            raise FileNotFoundError("Figure inaccessible.")
        document = allowed[figure["document_id"]]
        if document["sha256"] != figure["content_sha256"]:
            raise StaleFigureIndex("La version du document ne correspond plus à la figure.")
        self.engine.pdf_path(figure["document_id"], identity)
        path = (self.settings.root / figure["image_path"]).resolve()
        if not path.is_relative_to((self.settings.root / "data/figures_v2").resolve()):
            raise FileNotFoundError("Figure inaccessible.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != figure["image_sha256"]:
            raise StaleFigureIndex("Le fichier de la figure a changé : réindexation nécessaire.")
        return path


class MultimodalExplorer:
    def __init__(self, engine, settings, encoder=None):
        self.engine = engine
        self.figures = FigureStore(engine, settings, encoder)

    def close(self):
        self.figures.close()

    def search(self, request, identity):
        image = clean_image(request.image_base64)[0] if request.image_base64 else None
        with self.engine._lock:
            documents = self.engine.documents(identity)
            reference = (request.equipment_reference or "").strip().casefold()
            if reference:
                documents = [d for d in documents if reference in {str(r).casefold() for r in d.get("equipment_ids", [])}]
            document_ids = [d["id"] for d in documents]
            self.figures._check_snapshot()
            if not document_ids:
                return MultimodalResult(pages=[], figures_index_status=self.figures.status)
            context = self.engine._retrieve_text({"question": request.question, "identity": identity,
                                             "document_ids": document_ids})
            rankings = {"text": context["hits"]}
            rankings.update(self.figures.search(request.question, image, identity, document_ids))
            pages = fuse_rankings(rankings, limit=24)
            if self.engine.reranker.mode != "off":
                texts = {}
                for hits in rankings.values():
                    for hit in hits:
                        key = (hit["document_id"], hit["version"], hit["page"])
                        if hit.get("text", "").strip():
                            texts.setdefault(key, []).append(hit["text"])
                slots, passages = [], []
                for i, page in enumerate(pages):
                    text = "\n".join(dict.fromkeys(texts.get((page.document_id, page.version, page.page), [])))
                    if text:
                        slots.append(i)
                        passages.append(text[:12000])
                scores = self.engine.reranker.scores(request.question, passages) if passages else []
                ordered = [pages[slots[i]] for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))]
                # Les figures sans texte restent à leur rang RRF, sans légende inventée.
                for slot, page in zip(slots, ordered):
                    pages[slot] = page
            return MultimodalResult(pages=pages[:request.top_k],
                                    figures_index_status=self.figures.status)
