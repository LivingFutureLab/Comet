#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/10/30 10:56:28
# Author: Shilei Liu
SIMPLE_CHAT_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}{% generation %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% endgeneration %}<|im_end|>
{% else %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
