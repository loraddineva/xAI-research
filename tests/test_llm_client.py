"""
tests/test_llm_client.py
Tests for the LLM client using mocked provider responses.

Run with:
    pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.generation.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _model_cfg(
    model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    inference_provider: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        id="llama3-70b",
        provider="huggingface",
        model_name=model_name,
        max_tokens=128,
        temperature=0.0,
        inference_provider=inference_provider,
    )


EXPECTED_TEXT = "This is the model's response."


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

class TestHuggingFace:
    def test_generate_returns_text(self):
        mock_choice = MagicMock()
        mock_choice.message.content = f"  {EXPECTED_TEXT}  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        with patch(
            "huggingface_hub.InferenceClient",
            return_value=mock_client,
        ):
            client = LLMClient()
            result = client.generate("Test prompt", _model_cfg())

        assert result == EXPECTED_TEXT
        mock_client.chat_completion.assert_called_once()

    def test_passes_correct_params(self):
        mock_choice = MagicMock()
        mock_choice.message.content = EXPECTED_TEXT
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        model_cfg = _model_cfg("meta-llama/Meta-Llama-3-70B-Instruct")

        with patch(
            "huggingface_hub.InferenceClient",
            return_value=mock_client,
        ):
            client = LLMClient()
            client.generate("Hello", model_cfg)

        call_kwargs = mock_client.chat_completion.call_args.kwargs
        assert call_kwargs["model"] == "meta-llama/Meta-Llama-3-70B-Instruct"
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert "provider" not in call_kwargs

    def test_passes_inference_provider_when_set(self):
        mock_choice = MagicMock()
        mock_choice.message.content = EXPECTED_TEXT
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        model_cfg = _model_cfg(inference_provider="novita")

        with patch(
            "huggingface_hub.InferenceClient",
            return_value=mock_client,
        ):
            client = LLMClient()
            client.generate("Hello", model_cfg)

        call_kwargs = mock_client.chat_completion.call_args.kwargs
        assert call_kwargs["provider"] == "novita"

    def test_uses_base_url_for_inference_endpoint(self):
        mock_choice = MagicMock()
        mock_choice.message.content = EXPECTED_TEXT
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        model_cfg = ModelConfig(
            id="llama3-70b",
            provider="huggingface",
            model_name="meta-llama/Meta-Llama-3-70B-Instruct",
            max_tokens=128,
            temperature=0.0,
            base_url="https://my-endpoint.example.com",
        )

        with patch(
            "huggingface_hub.InferenceClient",
            return_value=mock_client,
        ) as mock_inference_cls:
            client = LLMClient()
            client.generate("Hello", model_cfg)

        mock_inference_cls.assert_called_once()
        assert mock_inference_cls.call_args.kwargs["base_url"] == "https://my-endpoint.example.com"

    def test_uses_env_base_url_for_placeholder_endpoint(self):
        import os

        mock_choice = MagicMock()
        mock_choice.message.content = EXPECTED_TEXT
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response

        model_cfg = ModelConfig(
            id="mistral-7b",
            provider="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            max_tokens=128,
            temperature=0.0,
            generation=False,
            base_url="https://YOUR_ENDPOINT.endpoints.huggingface.cloud",
        )

        with patch.dict(
            os.environ,
            {"HF_MISTRAL_ENDPOINT_URL": "https://env-mistral.endpoint.hf.cloud"},
        ):
            with patch(
                "huggingface_hub.InferenceClient",
                return_value=mock_client,
            ) as mock_inference_cls:
                client = LLMClient()
                client.generate("Hello", model_cfg)

        assert (
            mock_inference_cls.call_args.kwargs["base_url"]
            == "https://env-mistral.endpoint.hf.cloud"
        )


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_provider_raises(self):
        client = LLMClient()
        bad_cfg = ModelConfig(
            id="bad",
            provider="unknown_provider",
            model_name="test",
            max_tokens=128,
            temperature=0.0,
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            client.generate("prompt", bad_cfg)

    def test_missing_huggingface_hub_package(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            client = LLMClient()
            with pytest.raises((ImportError, TypeError)):
                client.generate("prompt", _model_cfg())
