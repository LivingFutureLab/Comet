export PYTHONPATH=.:$PYTHONPATH

tokenizer=/tmp/Qwen/Qwen3-0.6B
data=/tmp/data/scolls_32k_chatml_train.jsonl
save_prefix=/tmp/comet_data/20251202_mixed_scrolls_32k

python workflows/train_pre_tokenize.py --tokenizer_path ${tokenizer} --data_file ${data} --seed 0 --save_dir ${save_prefix}_epoch_0
python workflows/train_pre_tokenize.py --tokenizer_path ${tokenizer} --data_file ${data} --seed 1 --save_dir ${save_prefix}_epoch_1
python workflows/train_pre_tokenize.py --tokenizer_path ${tokenizer} --data_file ${data} --seed 2 --save_dir ${save_prefix}_epoch_2