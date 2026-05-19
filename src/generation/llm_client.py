"""
src/generation/llm_client.py
Unified LLM client that dispatches to the correct provider based on ModelConfig.

Usage
-----
    from src.generation.llm_client import LLMClient
    from src.config import load_config

    cfg = load_config()
    client = LLMClient()
    text = client.generate(prompt="Explain this.", model_cfg=cfg.get_model("llama3-70b"))

Providers
---------
    huggingface — Hugging Face Inference Providers / Endpoints via huggingface_hub
"""

from __future__ import annotations

import os
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import ModelConfig, resolve_model_base_url

# ---------------------------------------------------------------------------
# Retry policy (shared across all providers)
# ---------------------------------------------------------------------------

_RETRY_KWARGS: dict[str, Any] = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Provider-agnostic text generation client.

    Each provider method instantiates the SDK client once per generate() call,
    then wraps only the API call itself in the retry loop. Retries reuse the
    same client object rather than reconstructing it on every attempt.
    """

    def generate(
        self,
        prompt: str,
        model_cfg: ModelConfig,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a completion for *prompt* using the model specified in *model_cfg*.

        Args:
            prompt:     The full prompt string (system + user content merged).
            model_cfg:  A ModelConfig instance from AppConfig.models.
            max_tokens: Optional override for ``model_cfg.max_tokens`` (e.g. CoT runs).
            temperature: Optional override for ``model_cfg.temperature`` (e.g. robustness runs).

        Returns:
            The model's response as a plain string (stripped of leading/trailing
            whitespace).

        Raises:
            ValueError:  Unknown provider.
            Exception:   Provider API errors after all retries exhausted.
        """
        tokens = max_tokens if max_tokens is not None else model_cfg.max_tokens
        temp = temperature if temperature is not None else model_cfg.temperature
        provider = model_cfg.provider.lower()
        dispatch = {
            "huggingface": self._generate_huggingface,
        }
        if provider not in dispatch:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Valid options: {list(dispatch.keys())}"
            )
        return dispatch[provider](prompt, model_cfg, tokens, temp)

    # ------------------------------------------------------------------
    # Hugging Face (Inference Providers / Endpoints)
    # ------------------------------------------------------------------

    def _generate_huggingface(
        self, prompt: str, model_cfg: ModelConfig, max_tokens: int, temperature: float
    ) -> str:
        try:
            from huggingface_hub import InferenceClient as _InferenceClient
        except ImportError as e:
            raise ImportError(
                "huggingface_hub package not installed. Run: pip install huggingface_hub"
            ) from e

        client_kwargs: dict[str, Any] = {"token": _hf_token()}
        base_url = resolve_model_base_url(model_cfg)
        if base_url:
            client_kwargs["base_url"] = base_url
        client = _InferenceClient(**client_kwargs)

        completion_kwargs: dict[str, Any] = {
            "model": model_cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model_cfg.inference_provider:
            completion_kwargs["provider"] = model_cfg.inference_provider

        @retry(retry=retry_if_exception_type(Exception), **_RETRY_KWARGS)
        def _call() -> str:
            response = client.chat_completion(**completion_kwargs)
            return response.choices[0].message.content.strip()

        return _call()
