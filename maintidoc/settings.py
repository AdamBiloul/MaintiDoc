import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    root: Path = ROOT
    threshold: float = 0.40
    ollama_url: str = "http://127.0.0.1:11435"
    generation_model: str = "qwen2.5:1.5b"
    reranker_mode: str = "initial"
    reranker_adapter: str | None = None

    @classmethod
    def from_env(cls):
        value = cls(threshold=float(os.getenv("MAINTIDOC_THRESHOLD", "0.40")),
                    ollama_url=os.getenv("MAINTIDOC_OLLAMA_URL", "http://127.0.0.1:11435"),
                    generation_model=os.getenv("MAINTIDOC_LLM", "qwen2.5:1.5b"),
                    reranker_mode=os.getenv("MAINTIDOC_RERANKER", "initial"),
                    reranker_adapter=os.getenv("MAINTIDOC_RERANKER_ADAPTER"))
        if not 0 <= value.threshold <= 1:
            raise ValueError("MAINTIDOC_THRESHOLD doit être compris entre 0 et 1.")
        if value.reranker_mode not in {"off", "initial", "adapte"}:
            raise ValueError("MAINTIDOC_RERANKER : off, initial ou adapte.")
        return value
