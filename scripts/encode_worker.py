"""Processus éphémère BGE-M3 pour le profil CPU à mémoire limitée."""
import json
import sys

import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)
texts = json.load(sys.stdin)
model = SentenceTransformer(sys.argv[1], device="cpu", model_kwargs={"dtype": torch.float16})
vectors = model.encode(texts, normalize_embeddings=True, batch_size=1)
print(json.dumps(vectors.astype("float32").tolist()))
