"""VLM isolé : ne conserve pas les photos et libère toute sa mémoire en quittant."""
import base64
from io import BytesIO
import json
import sys

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

torch.set_num_threads(2)
request = json.load(sys.stdin)
image = Image.open(BytesIO(base64.b64decode(request["image"]))).convert("RGB")
processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    sys.argv[1], local_files_only=True, dtype=torch.float32,
    attn_implementation="eager",
).eval()
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Can you describe this image?"},
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
observation = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
if not observation.rstrip().endswith((".", "!", "?")):
    last_end = max(observation.rfind("."), observation.rfind("!"), observation.rfind("?"))
    if last_end >= 30:
        observation = observation[:last_end + 1]
print(json.dumps({"observation": observation}))
