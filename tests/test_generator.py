"""
tests/test_generator.py
Tests for generation model filtering (extraction-only models excluded).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from unittest.mock import patch

import pytest

from src.config import AppConfig, ModelConfig, resolve_model_base_url, validate_extraction_model


def _dual_model_config() -> AppConfig:
    return AppConfig(
        models=[
            ModelConfig(
                id="llama3-70b",
                provider="huggingface",
                model_name="meta-llama/Meta-Llama-3-70B-Instruct",
                generation=True,
            ),
            ModelConfig(
                id="mistral-7b",
                provider="huggingface",
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                generation=False,
                base_url="https://example.endpoints.huggingface.cloud",
            ),
        ],
    )


class TestGenerationModelFilter:
    def test_generation_models_excludes_extraction_only(self):
        cfg = _dual_model_config()
        gen_ids = [m.id for m in cfg.generation_models()]
        assert gen_ids == ["llama3-70b"]
        assert "mistral-7b" not in gen_ids

    def test_all_models_still_resolvable(self):
        cfg = _dual_model_config()
        assert cfg.get_model("mistral-7b").model_name == "mistralai/Mistral-7B-Instruct-v0.3"


class TestResolveModelBaseUrl:
    def test_generation_model_ignores_mistral_env_url(self):
        model = ModelConfig(
            id="llama3-70b",
            provider="huggingface",
            model_name="meta-llama/Meta-Llama-3-70B-Instruct",
            generation=True,
        )
        with patch.dict(
            os.environ,
            {"HF_MISTRAL_ENDPOINT_URL": "https://mistral.endpoint.hf.cloud"},
        ):
            assert resolve_model_base_url(model) is None

    def test_extraction_model_uses_mistral_env_url(self):
        model = ModelConfig(
            id="mistral-7b",
            provider="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            generation=False,
        )
        with patch.dict(
            os.environ,
            {"HF_MISTRAL_ENDPOINT_URL": "https://mistral.endpoint.hf.cloud"},
        ):
            assert resolve_model_base_url(model) == "https://mistral.endpoint.hf.cloud"


class TestValidateExtractionModel:
    def test_raises_when_endpoint_url_missing(self):
        model = ModelConfig(
            id="mistral-7b",
            provider="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            generation=False,
            base_url="https://YOUR_ENDPOINT.endpoints.huggingface.cloud",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_MISTRAL_ENDPOINT_URL", None)
            with pytest.raises(ValueError, match="Inference Endpoint"):
                validate_extraction_model(model)

    def test_passes_with_env_endpoint_url(self):
        model = ModelConfig(
            id="mistral-7b",
            provider="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            generation=False,
            base_url="https://YOUR_ENDPOINT.endpoints.huggingface.cloud",
        )
        with patch.dict(
            os.environ,
            {"HF_MISTRAL_ENDPOINT_URL": "https://test.endpoint.hf.cloud"},
        ):
            validate_extraction_model(model)
