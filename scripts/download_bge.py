"""Téléchargement BGE-M3 par blocs lorsque le transfert standard bloque."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
import urllib.request


REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
URL = f"https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}/pytorch_model.bin"
BASE_URL = f"https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}"
SIZE = 2271145830
SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
BLOCK = 8 * 1024 * 1024


def main():
    target = Path(".models/bge-m3/pytorch_model.bin")
    partial = target.with_suffix(".bin.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json", "config_sentence_transformers.json", "modules.json",
        "sentence_bert_config.json", "sentencepiece.bpe.model",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "1_Pooling/config.json",
    ):
        config = target.parent / name
        if not config.is_file():
            config.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
                data = response.read()
            config.write_bytes(data)
    if target.exists():
        print("Le fichier de poids existe déjà.")
        return
    # Ce téléchargement redémarre de zéro ; le fichier final n'est publié
    # qu'après réception de tous les blocs et vérification SHA-256.
    with partial.open("wb") as output:
        output.truncate(SIZE)

    def fetch(index):
        start = index * BLOCK
        end = min(start + BLOCK, SIZE) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    f"{URL}?block={index}",
                    headers={"Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Le serveur n'a pas respecté la plage demandée.")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("Bloc incomplet.")
                with partial.open("r+b") as output:
                    output.seek(start)
                    output.write(data)
                return len(data)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    count = (SIZE + BLOCK - 1) // BLOCK
    total = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for number, future in enumerate(as_completed(futures), start=1):
            total += future.result()
            if number % 8 == 0 or number == count:
                print(f"{total / SIZE:.0%} ({number}/{count} blocs)", flush=True)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("Empreinte SHA-256 incorrecte : fichier non activé.")
    os.replace(partial, target)
    print("Poids BGE-M3 téléchargés et empreinte SHA-256 vérifiée.")


if __name__ == "__main__":
    main()
