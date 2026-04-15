#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/12 22:33:23
# Author: Shilei Liu
# Adapted from
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


__all__ = ["GenerationArguments"]


@dataclass
class GenerationArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """

    use_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."
        },
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use sampling, use greedy decoding otherwise."
        },
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k filtering."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={
            "help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "The parameter for repetition penalty. 1.0 means no penalty."
        },
    )
    length_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length that is used with beam-based generation."
        },
    )
    num_return_sequences: int = field(
        default=1,
        metadata={
            "help": "The number of independently computed returned sequences for each element in the batch."
        },
    )

    def __post_init__(self):
        if not self.do_sample:
            self.top_p = None
            self.top_k = None
            self.temperature = None

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        return args

    def to_vllm_sampling_params_dict(self) -> Dict[str, Any]:
        args = {}
        if not self.do_sample:
            args["temperature"] = 0.0
        else:
            args["temperature"] = self.temperature
            args["top_p"] = self.top_p
            args["top_k"] = self.top_k
            args["repetition_penalty"] = self.repetition_penalty
        args["max_tokens"] = self.max_new_tokens
        return args
