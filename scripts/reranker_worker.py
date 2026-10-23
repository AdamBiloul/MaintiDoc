"""Inférence locale : entrée/sortie JSON, jamais de chargement de code distant."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maintidoc.reranker import verify_base

verify_base(Path(__file__).resolve().parents[1])

import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(2)
request = json.load(sys.stdin)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1], local_files_only=True)
if request.get("adapter"):
    model.classifier.load_state_dict(load_file(str(Path(request["adapter"]) / "adapter.safetensors")))
model.eval()
scores = []
with torch.inference_mode():
    for start in range(0, len(request["passages"]), 4):
        passages = request["passages"][start:start + 4]
        features = tokenizer([request["query"]] * len(passages), passages, padding=True,
                             truncation=True, max_length=384, return_tensors="pt")
        scores.extend(model(**features).logits.flatten().tolist())
print(json.dumps({"scores": scores}))
