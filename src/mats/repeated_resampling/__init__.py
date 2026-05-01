"""
Repeated resampling w/ Fireworks completions API. 

Resamples the final assistant turn one sentence at a time using stop sequences (["\n", ".\n", ". "]), and accepts/rejects sentences using LLM judge. This should not lead to tokenization drift, since tokens in the stop sequences are not merged with the final word in the sentence. 

This module is only suitable for rollouts that will end after the reasoning trace being resampled

NOTE: Currently only DeepSeek R1 (deepseek/deepseek-r1-0528) is working reliably.
Other models (v3.2, Kimi) have tokenizer issues that need to be resolved
"""

from .core import repeated_resampling, ResamplingResult
from .models import MODEL_REGISTRY, ModelConfig
from .utils import split_text_to_sentences, extract_last_assistant_reasoning

__all__ = [
    "repeated_resampling",
    "ResamplingResult",
    "MODEL_REGISTRY",
    "ModelConfig",
    "split_text_to_sentences",
    "extract_last_assistant_reasoning",
]
