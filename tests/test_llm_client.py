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
from src.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _model_cfg(provider: str, model_name: str = "test-model") -> ModelConfig:
    return ModelConfig(
        id=f"{provider}-test",
        provider=provider,
        model_name=model_name,
        max_tokens=128,
        temperature=0.0,
    )


EXPECTED_TEXT = "This is the model's response."


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestAnthropic:
    def test_generate_returns_text(self):
        mock_content = MagicMock()
        mock_content.text = f"  {EXPECTED_TEXT}  "
        mock_message = MagicMock()
        mock_message.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("anthropic.Anthropic", return_value=mock_client):
            client = LLMClient()
            result = client.generate("Test prompt", _model_cfg("anthropic"))

        assert result == EXPECTED_TEXT
        mock_client.messages.create.assert_called_once()

    def test_passes_correct_params(self):
        mock_content = MagicMock()
        mock_content.text = EXPECTED_TEXT
        mock_message = MagicMock()
        mock_message.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        model_cfg = _model_cfg("anthropic", "claude-opus-4-6")

        with patch("anthropic.Anthropic", return_value=mock_client):
            client = LLMClient()
            client.generate("Hello", model_cfg)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Together
# ---------------------------------------------------------------------------

class TestTogether:
    def test_generate_returns_text(self):
        mock_choice = MagicMock()
        mock_choice.message.content = f"  {EXPECTED_TEXT}  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("together.Together", return_value=mock_client):
            client = LLMClient()
            result = client.generate("Test prompt", _model_cfg("together"))

        assert result == EXPECTED_TEXT

    def test_passes_correct_params(self):
        mock_choice = MagicMock()
        mock_choice.message.content = EXPECTED_TEXT
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        model_cfg = _model_cfg("together", "meta-llama/Llama-3-70b-chat-hf")

        with patch("together.Together", return_value=mock_client):
            client = LLMClient()
            client.generate("Hello", model_cfg)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "meta-llama/Llama-3-70b-chat-hf"
        assert call_kwargs["max_tokens"] == 128


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

class TestMistral:
    def test_generate_returns_text(self):
        mock_choice = MagicMock()
        mock_choice.message.content = f"  {EXPECTED_TEXT}  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        with patch("mistralai.Mistral", return_value=mock_client):
            client = LLMClient()
            result = client.generate("Test prompt", _model_cfg("mistral"))

        assert result == EXPECTED_TEXT

    def test_passes_correct_params(self):
        mock_choice = MagicMock()
        mock_choice.message.content = EXPECTED_TEXT
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        model_cfg = _model_cfg("mistral", "mistral-small-latest")

        with patch("mistralai.Mistral", return_value=mock_client):
            client = LLMClient()
            client.generate("Hello", model_cfg)

        call_kwargs = mock_client.chat.complete.call_args.kwargs
        assert call_kwargs["model"] == "mistral-small-latest"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class TestOllama:
    def test_generate_returns_text(self):
        import json
        response_body = json.dumps({"message": {"content": f"  {EXPECTED_TEXT}  "}}).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = LLMClient()
            model_cfg = ModelConfig(
                id="ollama-test",
                provider="ollama",
                model_name="llama3:70b",
                max_tokens=128,
                temperature=0.0,
                base_url="http://localhost:11434",
            )
            result = client.generate("Test prompt", model_cfg)

        assert result == EXPECTED_TEXT


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_provider_raises(self):
        client = LLMClient()
        with pytest.raises(ValueError, match="Unknown provider"):
            client.generate("prompt", _model_cfg("unknown_provider"))

    def test_missing_anthropic_package(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            client = LLMClient()
            with pytest.raises((ImportError, TypeError)):
                client.generate("prompt", _model_cfg("anthropic"))
