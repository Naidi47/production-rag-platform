import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import config


class Embedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str | None = None):
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        if config.MODEL_DEVICE == "cuda":
            self.device = "cuda"
        elif config.MODEL_DEVICE == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        if self.dimension != config.PGVECTOR_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: model={self.dimension}, "
                f"PGVECTOR_DIMENSION={config.PGVECTOR_DIMENSION}"
            )
        self._initialized = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
