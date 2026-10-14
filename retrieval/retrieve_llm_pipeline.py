import os
from typing import Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None
_retrieve_llm_pipeline = None
_llm = None


def get_llm(
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.0,
    do_sample: bool = False,
    max_new_tokens: int = 512,
) -> HuggingFacePipeline:
    """Lazy-initializes and returns the HuggingFacePipeline LLM instance."""
    global _tokenizer, _model, _retrieve_llm_pipeline, _llm
    if _llm is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        _retrieve_llm_pipeline = pipeline(
            "text-generation",
            model=_model,
            tokenizer=_tokenizer,
            generation_config=generation_config,
            return_full_text=False,
        )
        _llm = HuggingFacePipeline(pipeline=_retrieve_llm_pipeline)
    return _llm


def __getattr__(name: str) -> Any:
    """Provides lazy access to module-level legacy exports without eager load."""
    if name == "llm":
        return get_llm()
    elif name == "retrieve_llm_pipeline":
        get_llm()
        return _retrieve_llm_pipeline
    elif name == "model_name":
        return DEFAULT_MODEL_NAME
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")