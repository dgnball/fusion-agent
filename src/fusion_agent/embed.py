"""Embedding models for text reports and imagery.

Two implementations share one interface:

- `LocalEmbedder` uses sentence-transformers. A MiniLM model embeds reports. CLIP embeds images and the
  text queries used to search them, so a query like "small boat with no lights" can match a picture.
- `HashEmbedder` needs no model at all. It is deterministic and used by tests and CI.
"""

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"


class Embedder(Protocol):
    name: str

    def text(self, texts: list[str]) -> np.ndarray: ...

    def image_query(self, texts: list[str]) -> np.ndarray: ...

    def images(self, paths: list[Path]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashEmbedder:
    """Bag-of-words hashing for text and a downsampled thumbnail for images."""

    name = "hash"
    dims = 256

    def text(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
                vectors[row, int.from_bytes(digest, "little") % self.dims] += 1
        return _normalise(vectors)

    def image_query(self, texts: list[str]) -> np.ndarray:
        return self.text(texts)

    def images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        vectors = [np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=np.float32).ravel() for p in paths]
        return _normalise(np.array(vectors))


class LocalEmbedder:
    name = "local"

    def __init__(self):
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer

        self._text = SentenceTransformer(TEXT_MODEL)
        self._clip = SentenceTransformer(IMAGE_MODEL)

    def text(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._text.encode(texts))

    def image_query(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._clip.encode(texts))

    def images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        return _normalise(self._clip.encode([Image.open(p).convert("RGB") for p in paths]))


def fetch_models() -> None:
    """Download model weights so later runs work with no network."""
    from sentence_transformers import SentenceTransformer

    for model in (TEXT_MODEL, IMAGE_MODEL):
        SentenceTransformer(model)


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown embedder {name}")
