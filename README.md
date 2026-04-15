# ☄️ CoMeT: Collaborative Memory Transformer
<sub>Official Implementation · [Paper](https://arxiv.org/abs/2602.01766) · MIT License</sub>

CoMeT is an efficient long-context LLM architecture that uses collaborative global and temporal memory mechanisms to process long sequences without quadratic attention costs.

![](assets/head.jpg)

## Requirements

- Python >= 3.10
- PyTorch 2.8.0
- NVIDIA H100 GPU or above
- [NVIDIA apex](https://github.com/NVIDIA/apex)

Install dependencies:
```bash
pip install -e .
```

## Quick Start

### 1. Download model and data

Download a supported backbone model from Hugging Face:
- [`Qwen/Qwen3-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)

Download the SCROLLS dataset (pre-processed for CoMeT):
- [Hugging Face](https://huggingface.co/datasets/OpenStellarTeam/scrolls_by_comet)
- [Kaggle](https://www.kaggle.com/datasets/liulangmingliu/scrolls-by-comet)

### 2. Pre-tokenize and pack data

```bash
bash scripts/tokenize.sh
```

### 3. Train

```bash
bash scripts/train.sh
```

### 4. Inference

```bash
bash scripts/infer.sh
```

If you don't have a trained model, you can use our pre-trained checkpoints below.


## Available Models

The CoMeT model trained on the SCROLLS dataset is available for download and testing. We host it on Kaggle due to upload speed limitations from mainland China to HuggingFace:

[Comet-Qwen3-4b-on-Scrolls](https://www.kaggle.com/models/liulangmingliu/comet-qwen3-4b-on-scrolls)


## Custom Data

To train on your own data, prepare the dataset in the following format:

- **File format**: JSONL (one JSON object per line)
- **Required fields**:
  - `uniq_id` (string): a unique identifier for each sample
  - `messages` (string): conversation data in ChatML format

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhao2026cometcollaborativememorytransformer,
      title={CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling},
      author={Runsong Zhao and Shilei Liu and Jiwei Tang and Langming Liu and Haibin Chen and Weidong Zhang and Yujin Yuan and Tong Xiao and Jingbo Zhu and Wenbo Su and Bo Zheng},
      year={2026},
      eprint={2602.01766},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.01766},
}
```

## License

This project is released under the [MIT License](LICENSE).
