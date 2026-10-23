"""Données d'adaptation et métriques communes aux trois classements.

Les candidats sont figés : aucun positif n'est ajouté pendant l'évaluation.
Une requête sans positif retrouvé reçoit zéro, même si son gold est connu ailleurs.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def validate_dataset(data, *, demo=False, root=None):
    if data.get("schema_version") != 1 or not isinstance(data.get("queries"), list):
        raise ValueError("Format de jeu de données invalide.")
    if not isinstance(data.get("demo_only"), bool) or data["demo_only"] != demo:
        raise ValueError("Un jeu synthétique exige --demo ; aucun entraînement métier implicite.")
    seen, groups, passages, counts = set(), {}, {}, Counter()
    source_cache = {}
    for query in data["queries"]:
        qid, split, group = query.get("id"), query.get("split"), query.get("group")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Identifiant de requête vide ou dupliqué.")
        seen.add(qid)
        if split not in {"train", "dev", "test"} or not isinstance(group, str) or not group:
            raise ValueError("Split ou groupe manquant.")
        if group in groups and groups[group] != split:
            raise ValueError("Fuite de groupe entre splits.")
        groups[group] = split
        question = query.get("query")
        if not isinstance(question, str) or not 3 <= len(question.strip()) <= 2000:
            raise ValueError("Question invalide.")
        question_hash = fingerprint(question.strip().casefold())
        if question_hash in passages and passages[question_hash] != split:
            raise ValueError("Question répétée entre splits.")
        passages[question_hash] = split
        candidates = query.get("candidates")
        gold = query.get("gold")
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 100:
            raise ValueError("Chaque requête exige 2 à 100 candidats figés.")
        if (not isinstance(gold, dict) or not gold or
                any(not isinstance(k, str) or type(v) is not int or not 0 <= v <= 3 for k, v in gold.items()) or
                not any(gold.values())):
            raise ValueError("Gold complet requis, pertinences entières entre 0 et 3.")
        ids, labels = set(), set()
        for candidate in candidates:
            cid, text, score = candidate.get("id"), candidate.get("text"), candidate.get("retrieval_score")
            if not isinstance(cid, str) or not cid or cid in ids or cid not in gold:
                raise ValueError("Candidat dupliqué ou sans annotation gold.")
            ids.add(cid)
            if not isinstance(text, str) or not 10 <= len(text.strip()) <= 12000:
                raise ValueError("Passage invalide.")
            if type(score) not in (float, int) or not math.isfinite(score):
                raise ValueError("Score de recherche non fini.")
            key = fingerprint(text.strip().casefold())
            if key in passages and passages[key] != split:
                raise ValueError("Passage répété entre splits.")
            passages[key] = split
            labels.add(gold[cid] > 0)
            counts[split] += 1
            if not demo:
                verify_source(candidate.get("source"), text, root, source_cache)
                document_key = "document:" + candidate["source"]["document_id"]
                if document_key in groups and groups[document_key] != split:
                    raise ValueError("Même document dans plusieurs splits : fuite documentaire.")
                groups[document_key] = split
        if split == "train" and labels != {False, True}:
            raise ValueError("L'entraînement exige des positifs et des négatifs par requête.")
        if not demo:
            review = query.get("review", {})
            if review.get("verified") is not True or not str(review.get("reviewer", "")).strip():
                raise ValueError("Validation humaine nominative manquante.")
            try:
                stamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    raise ValueError()
            except (ValueError, KeyError, AttributeError, TypeError):
                raise ValueError("Date de revue ISO avec fuseau requise.")
    if any(not counts[s] for s in ("train", "dev", "test")):
        raise ValueError("Trois splits non vides requis.")
    total = sum(counts.values())
    if not demo and not 800 <= total <= 1500:
        raise ValueError(f"Le livrable exige 800 à 1500 paires vérifiées ; reçu : {total}.")
    return {"queries": len(seen), "pairs": total, "split_pairs": dict(counts),
            "demo_only": demo, "dataset_sha256": fingerprint(data)}


def verify_source(source, text, root, cache):
    if root is None or not isinstance(source, dict):
        raise ValueError("Racine du corpus et provenance PDF requises.")
    for field in ("document_id", "version", "path", "sha256", "page"):
        if not source.get(field):
            raise ValueError("Provenance documentaire incomplète.")
    if "_documents" not in cache:
        state = json.loads((Path(root) / "data/index_v1.json").read_text(encoding="utf-8"))
        cache["_documents"] = {d["id"]: d for d in state["documents"] if d.get("authorized") is True}
    document = cache["_documents"].get(source["document_id"])
    if document is None or any(source[k] != document[k] for k in ("version", "path", "sha256")):
        raise ValueError("La source ne correspond pas au corpus autorisé actif.")
    path = (Path(root) / source["path"]).resolve()
    if not path.is_relative_to((Path(root) / "data/raw").resolve()):
        raise ValueError("Source en dehors de data/raw.")
    if path not in cache:
        import pymupdf
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("L'empreinte du PDF source a changé.")
        with pymupdf.open(path) as pdf:
            cache[path] = {"sha256": source["sha256"], "pages": [" ".join(p.get_text().split()) for p in pdf]}
    if cache[path]["sha256"] != source["sha256"]:
        raise ValueError("Empreintes contradictoires pour la même source.")
    if type(source["page"]) is not int or not 1 <= source["page"] <= len(cache[path]["pages"]):
        raise ValueError("Page source invalide.")
    if " ".join(text.split()) not in cache[path]["pages"][source["page"] - 1]:
        raise ValueError("Le passage n'est pas vérifiable dans la page PDF.")


def ranking_metrics(order, gold):
    if len(set(order)) != len(order) or any(cid not in gold for cid in order):
        raise ValueError("Classement dupliqué ou inconnu.")
    dcg = sum((2 ** gold[cid] - 1) / math.log2(i + 2) for i, cid in enumerate(order[:5]))
    ideal = sum((2 ** rel - 1) / math.log2(i + 2)
                for i, rel in enumerate(sorted(gold.values(), reverse=True)[:5]))
    mrr = next((1 / (i + 1) for i, cid in enumerate(order[:10]) if gold[cid] > 0), 0.0)
    return {"ndcg@5": dcg / ideal if ideal else 0.0, "mrr@10": mrr}


def compare_rankings(queries, scorer):
    records = []
    for q in queries:
        baseline = sorted(q["candidates"], key=lambda c: (-c["retrieval_score"], c["id"]))
        ids = [c["id"] for c in baseline]
        modes = {"sans_reranker": ids}
        for mode in ("initial", "adapte"):
            scores = scorer(mode, q["query"], [c["text"] for c in baseline])
            if len(scores) != len(ids) or not all(math.isfinite(v) for v in scores):
                raise ValueError("Sortie de reranker invalide.")
            modes[mode] = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))]
        records.append({"id": q["id"], "candidate_ids": ids,
                        "rankings": modes, "metrics": {m: ranking_metrics(v, q["gold"]) for m, v in modes.items()}})
    if not records:
        raise ValueError("Split d'évaluation vide.")
    means = {mode: {metric: sum(r["metrics"][mode][metric] for r in records) / len(records)
                    for metric in ("ndcg@5", "mrr@10")}
             for mode in ("sans_reranker", "initial", "adapte")}
    return {"queries": len(records), "means": means, "details": records,
            "positives_injected": False, "candidate_protocol": "identical_frozen_candidates"}
