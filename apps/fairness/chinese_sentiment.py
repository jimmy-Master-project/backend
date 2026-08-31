from __future__ import annotations

from pathlib import Path
from typing import Any


class XlmRobertaSentimentClassifier:
    """LangFair adapter for CardiffNLP's XLM-R multilingual sentiment model."""

    def __init__(self, device: str = "cpu") -> None:
        from transformers import pipeline

        device_id = 0 if device == "cuda" else -1
        model_path = Path(__file__).resolve().parents[2] / "models" / "xlm-roberta-sentiment"
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Local XLM-R sentiment model is missing: {model_path}"
            )
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device_id,
        )

    def predict(self, texts: list[str]) -> list[float]:
        results: list[Any] = self._pipeline(texts, top_k=3, truncation=True)
        scores: list[float] = []
        for result in results:
            labels = result if isinstance(result, list) else [result]
            positive = next(
                (item["score"] for item in labels if str(item["label"]).lower() in {"positive", "label_2"}),
                0.0,
            )
            scores.append(float(positive))
        return scores
