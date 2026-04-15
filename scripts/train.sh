export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_DEBUG="WARN"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.:$PYTHONPATH
export OTEL_SDK_DISABLED=true

table_0="/tmp/comet_data/20251202_mixed_scrolls_32k_epoch_0"
table_1="/tmp/comet_data/20251202_mixed_scrolls_32k_epoch_1"
table_2="/tmp/comet_data/20251202_mixed_scrolls_32k_epoch_2"
tables="${table_0},${table_1},${table_2}"

ckpt_dir="/tmp/comet/qwen3_4b_on_scrolls_32k"
model_path="/tmp/Qwen/Qwen3-0.6B"

torchrun --nproc_per_node 8 --master-port 29502 workflows/train_from_qwen3.py \
--model_path=${model_path} \
--save=${ckpt_dir} \
--load=${ckpt_dir} \
--timing_log_level=1 \
--data_source=chunk \
--tables=${tables} \
--input_ids_col input_ids \
--labels_col labels \
--max_seq_length 32768 \
--chunk_size 2048 \
--global_mem_size 512 \
--temp_beacon_stride 8 \
--temp_mem_budget 2048 \
--pp_size 1 \
--zero3 true \
--recompute true \
--micro_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--log_interval 10 \
--save_interval 200 \
--warmup_steps 10 \
--max_lr 5e-5 \
--min_lr 0.0 \
--scheduler cosine \
--beta1 0.9 \
--beta2 0.999 \
--weight_decay 0.0 \
--dtype bf16