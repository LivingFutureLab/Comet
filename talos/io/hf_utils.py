#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/23 20:55:45
# Author: Shilei Liu

HF_CKPT_PATTERNS = [
    "*pytorch_model.bin",
    "*pytorch_model-*-of-*.bin",
    "*model-*-of-*.safetensors",
    "*model.safetensors",
    "*pytorch_model.bin.index.json",
    "*model.safetensors.index.json",
    "*config.json",
    "*special_tokens_map.json",
    "*tokenizer_config.json",
    "*vocab.json",
    "*tokenizer.json",
    "*merges.txt",
    "*generation_config.json",
    "*configuration_*.py",
    "*modeling_*.py",
    "*tokenization_*.py",
    "*chat_template.jinja",
    "*added_tokens.json",
    "*preprocessor_config.json",
    "*video_preprocessor_config.json",
]
