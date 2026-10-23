"""Adapte uniquement la tête de classement ; encodeur gelé pour cette machine CPU.

Sans --demo, exige 800–1500 paires validées avec provenance PDF contrôlée.
Le test reste fermé jusqu'au checkpoint choisi sur dev puis sauvegardé.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
from maintidoc.adaptation import compare_rankings, ranking_metrics, validate_dataset
from maintidoc.reranker import verify_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 0 < args.learning_rate <= 0.01:
        parser.error("Epochs 1..100 ; learning-rate >0 et <=0.01.")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    validation = validate_dataset(dataset, demo=args.demo, root=ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / ".models" / ("reranker-demo-" + stamp if args.demo else "reranker-adapte-" + stamp)).resolve()
    if not output.is_relative_to((ROOT / ".models").resolve()) or output.exists():
        parser.error("Choisir un nouveau dossier dans .models ; aucun modèle existant n'est écrasé.")
    # Les erreurs de données sont détectées avant d'importer/charger les modèles.
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    started = time.perf_counter()
    base = ROOT / ".models/reranker-base"
    base_record = verify_base(ROOT)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    initial = {k: v.detach().clone() for k, v in model.classifier.state_dict().items()}
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train = [q for q in dataset["queries"] if q["split"] == "train"]
    dev = [q for q in dataset["queries"] if q["split"] == "dev"]
    test = [q for q in dataset["queries"] if q["split"] == "test"]

    def features(question, passages):
        return tokenizer([question] * len(passages), passages, truncation=True,
                         padding=True, max_length=384, return_tensors="pt")

    def predict(question, passages):
        model.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(passages), 4):
                result.extend(model(**features(question, passages[start:start + 4])).logits.flatten().tolist())
        return result

    history, best, best_metric, best_epoch, steps = [], None, -1.0, None, 0
    for epoch in range(args.epochs):
        random.shuffle(train)
        total_loss, batches = 0.0, 0
        model.eval()  # Encodeur et dropout fixes ; seule la tête reçoit des gradients.
        for query in train:
            candidates = query["candidates"]
            for start in range(0, len(candidates), 4):
                batch = candidates[start:start + 4]
                optimizer.zero_grad(set_to_none=True)
                logits = model(**features(query["query"], [c["text"] for c in batch])).logits.flatten()
                labels = torch.tensor([float(query["gold"][c["id"]] > 0) for c in batch])
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                batches += 1
                steps += 1
        dev_metrics = []
        for query in dev:
            candidates = query["candidates"]
            scores = predict(query["query"], [c["text"] for c in candidates])
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], candidates[i]["id"]))
            dev_metrics.append(ranking_metrics([candidates[i]["id"] for i in order], query["gold"]))
        metric = sum(m["ndcg@5"] for m in dev_metrics) / len(dev_metrics)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / batches, "dev_ndcg@5": metric})
        # En cas d'égalité, conserve le premier checkpoint, sans consulter test.
        if metric > best_metric:
            best_metric, best_epoch = metric, epoch + 1
            best = {k: v.detach().clone() for k, v in model.classifier.state_dict().items()}
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={total_loss / batches:.4f}, dev nDCG@5={metric:.4f}", flush=True)
    output.mkdir(parents=True)
    adapter_path = output / "adapter.safetensors"
    save_file(best, str(adapter_path))
    changed = any(not torch.equal(initial[k], best[k]) for k in initial)
    if not changed:
        raise RuntimeError("Les poids n'ont pas changé : adaptation non démontrée.")
    metadata = {"schema_version": 1, "created_at": stamp, "demo_only": args.demo,
                "base": base_record, "dataset": validation, "method": "frozen_encoder_classifier_head",
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "epochs": args.epochs, "optimizer_steps": steps, "selected_epoch": best_epoch,
                "learning_rate": args.learning_rate, "seed": 42, "max_length": 384,
                "weights_changed": changed, "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                "history": history, "checkpoint_selection": "dev_ndcg@5_only_first_best"}
    (output / "adaptation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def scorer(mode, question, passages):
        model.classifier.load_state_dict(initial if mode == "initial" else best)
        return predict(question, passages)

    # Rechargement effectif depuis le fichier pour comparer le checkpoint sauvegardé.
    from safetensors.torch import load_file
    best = load_file(str(adapter_path))
    comparison = compare_rankings(test, scorer)
    report = {"demo_only": args.demo, "validation": validation, "adapter": str(output.relative_to(ROOT)),
              "training": metadata, "test": comparison, "seconds": round(time.perf_counter() - started, 2),
              "notice": "Démonstration technique synthétique ; aucune validation métier." if args.demo else
                        "Évaluation sur candidats figés ; compléter la recette métier et les contrôles terrain."}
    report_path = ROOT / "data/evaluation" / ("v2_reranker_demo.json" if args.demo else f"v2_reranker_{stamp}.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "test_means": comparison["means"], "demo_only": args.demo}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Entraînement refusé : {error}")
