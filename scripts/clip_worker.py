"""Un seul encodeur par processus pour libérer sa mémoire avant le suivant."""
import base64
from io import BytesIO
import json
import sys

from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)
kind, path = sys.argv[1:]
items = json.load(sys.stdin)
model = SentenceTransformer(path, device="cpu")
if kind == "image":
    items = [Image.open(BytesIO(base64.b64decode(item))).convert("RGB") for item in items]
vectors = model.encode(items, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
print(json.dumps(vectors.astype("float32").tolist()))
