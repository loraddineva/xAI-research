"""
src/generation/llm_client.py
Unified LLM client that dispatches to the correct provider based on ModelConfig.

Usage
-----
    from src.generation.llm_client import LLMClient
    from src.config import load_config

    cfg = load_config()
    client = LLMClient()
    text = client.generate(prompt="Explain this.", model_cfg=cfg.get_model("claude-opus"))

Providers
---------
    anthropic   — Anthropic Messages API (claude-*)
    together    — Together AI Chat Completions (Llama etc.)
    mistral     — Mistral AI Chat Completions
    ollama      — Local Ollama HTTP API (stubbed; enable when local server is running)
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

from src.config import ModelConfig

# ---------------------------------------------------------------------------
# Retry policy (shared across all providers)
# ---------------------------------------------------------------------------

_RETRY_KWARGS: dict[str, Any] = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)


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
            "anthropic": self._generate_anthropic,
            "together":  self._generate_together,
            "mistral":   self._generate_mistral,
            "ollama":    self._generate_ollama,
        }
        if provider not in dispatch:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Valid options: {list(dispatch.keys())}"
            )
        return dispatch[provider](prompt, model_cfg, tokens, temp)

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _generate_anthropic(
        self, prompt: str, model_cfg: ModelConfig, max_tokens: int, temperature: float
    ) -> str:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e

        client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        @retry(retry=retry_if_exception_type(Exception), **_RETRY_KWARGS)
        def _call() -> str:
            message = client.messages.create(
                model=model_cfg.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()

        return _call()

    # ------------------------------------------------------------------
    # Together AI
    # ------------------------------------------------------------------

    def _generate_together(
        self, prompt: str, model_cfg: ModelConfig, max_tokens: int, temperature: float
    ) -> str:
        try:
            from together import Together as _Together
        except ImportError as e:
            raise ImportError("together package not installed. Run: pip install together") from e

        client = _Together(api_key=os.environ.get("TOGETHER_API_KEY"))

        @retry(retry=retry_if_exception_type(Exception), **_RETRY_KWARGS)
        def _call() -> str:
            response = client.chat.completions.create(
                model=model_cfg.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

        return _call()

    # ------------------------------------------------------------------
    # Mistral
    # ------------------------------------------------------------------

    def _generate_mistral(
        self, prompt: str, model_cfg: ModelConfig, max_tokens: int, temperature: float
    ) -> str:
        try:
            from mistralai.client import Mistral as _Mistral
        except ImportError as e:
            raise ImportError("mistralai package not installed. Run: pip install mistralai") from e

        client = _Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

        @retry(retry=retry_if_exception_type(Exception), **_RETRY_KWARGS)
        def _call() -> str:
            response = client.chat.complete(
                model=model_cfg.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

        return _call()

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------

    def _generate_ollama(
        self, prompt: str, model_cfg: ModelConfig, max_tokens: int, temperature: float
    ) -> str:
        """
        Calls the local Ollama REST API.
        Set base_url in the model config (default: http://localhost:11434).
        """
        import json
        import urllib.request

        base_url = (model_cfg.base_url or "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"

        payload = json.dumps({
            "model": model_cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        @retry(retry=retry_if_exception_type(Exception), **_RETRY_KWARGS)
        def _call() -> str:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()

        return _call()
