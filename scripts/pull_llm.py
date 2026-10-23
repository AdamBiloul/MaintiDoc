"""Télécharge le modèle de démonstration dans l'instance Ollama du projet."""
import json
import httpx

last = None
with httpx.stream("POST", "http://127.0.0.1:11435/api/pull",
                  json={"model": "qwen2.5:1.5b"}, timeout=600) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        item = json.loads(line)
        if "error" in item:
            raise RuntimeError(item["error"])
        progress = int(10 * item.get("completed", 0) / item["total"]) * 10 if item.get("total") else None
        current = (item.get("status"), progress)
        if current != last:
            print(item.get("status"), f"{progress}%" if progress is not None else "", flush=True)
            last = current
