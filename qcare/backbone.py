"""The LLM backbone that performs every Q-CARE judgement.

The backbone is a local HuggingFace causal LM. Swap it by passing a different
model id: any instruction-tuned chat model with a tokenizer chat template works.
A base (non instruction-tuned) model will not follow the judgement prompts.

Determinism: the pipeline requests greedy decoding (``temperature=0.0``) for
every step, so results are reproducible for a fixed model, library versions and
GPU. A seed is set for the sampling path.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_MAX_NEW_TOKENS = 2000


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Backbone:
    """Loads a local chat model and generates responses for prompt messages."""

    def __init__(self, model_name: str = DEFAULT_MODEL, seed: int = 1234):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        set_seed(seed)

    def load(self) -> "Backbone":
        """Load weights and tokenizer. Sharded across visible GPUs."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        return self

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> str:
        """Generate a response for chat-formatted messages.

        ``temperature=0.0`` means greedy decoding, which is what every pipeline
        step uses. Transient generation errors are retried a few times.
        """
        if self.model is None:
            raise RuntimeError("Backbone is not loaded; call .load() first.")

        if temperature == 0.0 or temperature is None:
            do_sample, temp, top_p = False, None, None
        else:
            do_sample, temp, top_p = True, temperature, 0.8

        for attempt in range(max_retries):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temp,
                    top_p=top_p,
                    top_k=None,
                )
                prompt_length = inputs.input_ids.shape[-1]
                return self.tokenizer.decode(
                    output_ids[0][prompt_length:], skip_special_tokens=True
                )
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                if attempt == max_retries - 1:
                    raise
                print(f"    generation error (attempt {attempt + 1}/{max_retries}): {exc}")
                time.sleep(1)
        return ""
