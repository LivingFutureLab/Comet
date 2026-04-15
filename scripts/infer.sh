export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.:$PYTHONPATH
export OTEL_SDK_DISABLED=true

model_path="<your checkpoint path>"
tables="/tmp/data/scolls_32k_chatml_eval.jsonl"

python workflows/inference.py \
--model_path=${model_path} \
--tables=${tables} \
--max_input_length 32256 \
--max_new_tokens 512 \
--dtype bf16