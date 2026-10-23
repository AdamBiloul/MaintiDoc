"""Lance une instance Ollama dédiée à MaintiDoc, sans modifier l'instance existante."""
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

root = Path(__file__).resolve().parents[1]
runtime = root / "data/runtime"
runtime.mkdir(parents=True, exist_ok=True)
try:
    with urllib.request.urlopen("http://127.0.0.1:11435/api/version", timeout=2):
        print("Ollama répond déjà sur 127.0.0.1:11435.")
except OSError:
    executable = shutil.which("ollama")
    if not executable:
        raise SystemExit("Installez Ollama : https://ollama.com/download")
    environment = dict(os.environ, OLLAMA_HOST="127.0.0.1:11435",
                       OLLAMA_MODELS=str(root / ".models/ollama"), OLLAMA_NO_CLOUD="1")
    with (runtime / "ollama.out.log").open("a") as out, (runtime / "ollama.err.log").open("a") as err:
        process = subprocess.Popen([executable, "serve"], env=environment, cwd=root,
                                   stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    (runtime / "ollama.pid").write_text(str(process.pid), encoding="ascii")
    print(f"Ollama lancé sur 127.0.0.1:11435 (PID {process.pid}).")
