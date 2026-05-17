import re
from typing import Any, Optional, Type

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel
import ollama


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


class OllamaJudge(DeepEvalBaseLLM):
    """Ollama-backed judge for DeepEval metrics.

    DeepEval calls ``generate_with_schema`` for faithfulness/relevancy steps.
    The base implementation falls through to plain ``generate`` because
    ``generate`` does not accept ``schema=``, so Ollama never saw a JSON schema.
    Here we pass ``format=<json_schema>`` so the server constrains output to JSON.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        return self

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": str(prompt)}],
        )
        return response["message"]["content"]

    def generate_with_schema(
        self,
        prompt: Any,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Any:
        if schema is None:
            return self.generate(str(prompt))
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return super().generate_with_schema(prompt, schema=schema, **kwargs)
        fmt = schema.model_json_schema()
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": str(prompt)}],
            format=fmt,
        )
        raw = _strip_json_fences(response["message"]["content"])
        return schema.model_validate_json(raw)

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt)

    async def a_generate_with_schema(
        self,
        prompt: Any,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Any:
        return self.generate_with_schema(prompt, schema=schema, **kwargs)

    def supports_json_mode(self):
        return True

    def get_model_name(self):
        return self.model_name