from sentence_transformers import CrossEncoder

from src.config import config


class Reranker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str | None = None):
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.model = CrossEncoder(self.model_name, max_length=config.RERANKER_MAX_LENGTH)
        self._initialized = True

    def rerank(self, query: str, chunks: list[str]) -> list[tuple[int, float]]:
        if not chunks:
            return []
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=config.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
        indexed = [(i, float(score)) for i, score in enumerate(scores)]
        indexed.sort(key=lambda item: item[1], reverse=True)
        return indexed

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
