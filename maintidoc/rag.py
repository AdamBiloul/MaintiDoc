"""Chaîne LangChain : recherche autorisée → génération → validation des références."""
import hashlib
import json
from pathlib import Path
import threading
import re
import unicodedata

import httpx
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient, models

from maintidoc.embeddings import BGE
from maintidoc.reranker import LocalReranker
from maintidoc.schemas import Answer, Citation, Claim, Draft, GenerationOutput


def terms(text):
    normalized = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    stop = {"quel", "quelle", "quels", "quelles", "les", "des", "une", "est", "sont", "dans", "pour", "avant", "avec", "qui", "que", "quoi", "par", "sur", "aux", "permettent", "doit"}
    return {word.rstrip("s") for word in re.findall(r"[a-z0-9_/+-]+", normalized) if len(word) > 2 and word not in stop}


def abstain(reason):
    return Answer(status="abstained", answer="Les documents accessibles ne permettent pas de fournir une réponse vérifiable. Précisez l'équipement ou ajoutez le manuel correspondant.", citations=[], reason=reason)


class OllamaGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.prompt = (settings.root / "prompts/rag_v1.txt").read_text(encoding="utf-8")

    def __call__(self, question, hits):
        context = []
        query_terms = terms(question)
        for source_id, hit in enumerate(hits, 1):
            normalized = " ".join(hit["text"].split())
            for sentence in re.split(r"(?<=[.!?;])\s+|(?=•)", normalized):
                words = sentence.split()
                while words:
                    count = min(70, len(words))
                    text = " ".join(words[:count])
                    while len(text) > 750 and count > 1:
                        count -= 1
                        text = " ".join(words[:count])
                    words = words[count:]
                    if 10 <= len(text) <= 800:
                        context.append({"source_id": source_id, "document": hit["document"],
                                        "page": hit["page"], "text": text,
                                        "rank_score": len(query_terms & terms(text)) + 0.25 * hit["score"] +
                                        (0.5 / (1 + hit["rerank_position"]) if "rerank_position" in hit else 0),
                                        "rerank_position": hit.get("rerank_position")})
        context.sort(key=lambda c: (-c["rank_score"],
                                    c["rerank_position"] if c["rerank_position"] is not None else 0))
        unique, seen = [], set()
        for item in context:
            key = (item["document"], item["page"], item["text"])
            if key not in seen:
                seen.add(key)
                unique.append({k: v for k, v in item.items() if k not in {"rank_score", "rerank_position"}})
        context = [dict(c, id=i) for i, c in enumerate(unique[:4], 1)]
        if not context:
            return Draft(supported=False, claims=[])
        schema = GenerationOutput.model_json_schema()
        schema["properties"]["evidence_ids"]["items"] = {"type": "integer", "enum": [c["id"] for c in context]}
        # Aucun outil n'est exposé au modèle ; une seule requête, sans boucle d'agent.
        response = httpx.post(f"{self.settings.ollama_url}/api/chat", timeout=180, json={
            "model": self.settings.generation_model, "stream": False,
            "messages": [{"role": "system", "content": self.prompt},
                         {"role": "user", "content": json.dumps({"question": question, "passages": context}, ensure_ascii=False)}],
            "format": schema,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 600},
            "keep_alive": 0,
        })
        response.raise_for_status()
        output = GenerationOutput.model_validate_json(response.json()["message"]["content"])
        choices = {c["id"]: c for c in context}
        claims = []
        for identifier in dict.fromkeys(output.evidence_ids):
            if identifier not in choices:
                raise ValueError("Identifiant de preuve inconnu.")
            choice = choices[identifier]
            claims.append(Claim(text=choice["text"], source_id=choice["source_id"], quote=choice["text"]))
        return Draft(supported=bool(claims), claims=claims)


def verified_answer(draft, hits):
    if not draft.supported:
        return abstain("informations_insuffisantes")
    citations, sentences = [], []
    for claim in draft.claims:
        if claim.source_id > len(hits):
            return abstain("citation_invalide")
        hit = hits[claim.source_id - 1]
        # Tolérance aux espaces, sans tolérance aux mots ou valeurs inventés.
        quote = " ".join(claim.quote.split())
        if quote not in " ".join(hit["text"].split()):
            return abstain("citation_invalide")
        label = f"S{claim.source_id}"
        # V1 extractive : aucune reformulation libre ne dépasse la preuve citée.
        sentences.append(f"« {quote} » [{label}]")
        citations.append(Citation(label=label, document_id=hit["document_id"],
                                  document=hit["document"], version=hit["version"],
                                  page=hit["page"], quote=quote, score=hit["score"]))
    return Answer(status="answered", answer="\n\n".join(sentences), citations=citations)


class RagEngine:
    def __init__(self, settings, embedder=None, generator=None, reranker=None):
        self.settings = settings
        self.state = json.loads((settings.root / "data/index_v1.json").read_text(encoding="utf-8"))
        self.client = QdrantClient(path=str(settings.root / "data/qdrant_v1"))
        self.embedder = embedder or BGE(settings.root)
        self.generator = generator or OllamaGenerator(settings)
        self.reranker = reranker or LocalReranker(settings.root, settings.reranker_mode, settings.reranker_adapter)
        self.figure_store = None
        self._lock = threading.Lock()
        self.chain = RunnableLambda(self._retrieve) | RunnableLambda(self._generate) | RunnableLambda(self._validate)

    def close(self):
        self.client.close()

    def documents(self, identity):
        roles = set(identity.roles)
        return [d for d in self.state["documents"] if roles.intersection(d["allowed_roles"])]

    def pdf_path(self, document_id, identity):
        document = next((d for d in self.documents(identity) if d["id"] == document_id), None)
        if document is None:
            raise FileNotFoundError("Document inaccessible.")
        path = (self.settings.root / document["path"]).resolve()
        if not path.is_relative_to((self.settings.root / "data/raw").resolve()):
            raise FileNotFoundError("Document inaccessible.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != document["sha256"]:
            raise ValueError("Le document a changé : réindexez-le avant de consulter ses références.")
        return path

    def ask(self, question, identity):
        # Qdrant local et les modèles sont utilisés dans un seul workflow à la fois.
        with self._lock:
            return self.chain.invoke({"question": question, "identity": identity})

    def _retrieve(self, context):
        context = self._retrieve_text(context)
        # Une photo peut modifier l'ordre des pages, jamais fournir seule une preuve.
        if context.get("visual_query") is not None and self.figure_store is not None and context.get("_query_vector") is not None:
            from maintidoc.multimodal import fuse_rankings
            document_ids = context.get("document_ids", [d["id"] for d in self.documents(context["identity"])])
            rankings = {"text": context["hits"]}
            rankings.update(self.figure_store.search(context["question"], context["visual_query"],
                                                     context["identity"], document_ids))
            pages = fuse_rankings(rankings, limit=5)
            # Récupère les vrais passages des pages visuelles, avec les mêmes ACL.
            # Un cosinus CLIP ne devient jamais un score de pertinence documentaire.
            candidates = {h["id"]: h for h in context["hits"]}
            for page in pages:
                filters = models.Filter(must=[
                    models.FieldCondition(key="allowed_roles", match=models.MatchAny(any=context["identity"].roles)),
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=page.document_id)),
                    models.FieldCondition(key="version", match=models.MatchValue(value=page.version)),
                    models.FieldCondition(key="page", match=models.MatchValue(value=page.page)),
                ])
                points, _ = self.client.scroll(self.state["collection"], scroll_filter=filters,
                                               limit=32, with_payload=True, with_vectors=True)
                for point in points:
                    score = float(sum(a * b for a, b in zip(context["_query_vector"], point.vector)))
                    if score >= self.settings.threshold:
                        candidates[point.payload["id"]] = dict(point.payload, score=score)
            ranks = {(p.document_id, p.version, p.page): i for i, p in enumerate(pages)}
            hits = sorted(candidates.values(), key=lambda h: ranks.get((h["document_id"], h["version"], h["page"]), 999))
            context = {**context, "hits": hits[:8]}
        return {**context, "hits": self.reranker.rerank(context["question"], context["hits"])}

    def _retrieve_text(self, context):
        if not self.documents(context["identity"]):
            return {**context, "hits": []}
        vector = self.embedder.encode([context["question"]])[0]
        conditions = [models.FieldCondition(key="allowed_roles", match=models.MatchAny(any=context["identity"].roles))]
        if "document_ids" in context:
            if not context["document_ids"]:
                return {**context, "hits": []}
            conditions.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=context["document_ids"])))
        hits = self.client.query_points(
            collection_name=self.state["collection"], query=vector.tolist(), limit=8,
            query_filter=models.Filter(must=conditions), with_payload=True,
        ).points
        if not hits or hits[0].score < self.settings.threshold:
            hits = []
        return {**context, "_query_vector": vector.tolist(), "hits": [dict(h.payload, score=float(h.score)) for h in hits]}

    def _generate(self, context):
        if not context["hits"]:
            return {**context, "draft": None}
        try:
            draft = self.generator(context["question"], context["hits"])
        except (ValueError, KeyError, TypeError):
            draft = None
        return {**context, "draft": draft}

    def _validate(self, context):
        if not context["hits"]:
            return abstain("aucune_source_accessible_pertinente")
        if context["draft"] is None:
            return abstain("generation_non_verifiable")
        return verified_answer(context["draft"], context["hits"])
