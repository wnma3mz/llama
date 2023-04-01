# LLaMA 

本项目基于[LLaMA](https://github.com/facebookresearch/llama)修改而来。出于简单和学习的目的，本项目坚持使用一些易安装、易使用的第三方库（即不使用Transformer、PyTorch Lightning）。

主要新增内容如下：

- 7B 模型最低GPU显存要求14G，这对于小显存的显卡并不友好。这里依据原项目拆分为四个模型。以支持在四张较小显存的显卡机器上并行运行。注：实际总的占用显存会更大

模型拆分后的[下载地址](https://huggingface.co/wnma3mz/llama_fs_7B/tree/main)

下一步计划，在四张显卡上并行微调。

- 参考项目[Peft](https://github.com/huggingface/peft)


This project is based on a modification of [LLaMA](https://github.com/facebookresearch/llama). For simplicity and learning purposes, this project sticks to some easy-to-install and easy-to-use third-party libraries (i.e. no Transformer, PyTorch Lightning).

The main additions are as follows:

- 7B models require a minimum GPU video memory of 14G, which is not friendly to small video memory cards. Here it is split into four models based on the original project. This is to support running on multi-card machines with smaller graphics memory. Note: the actual total video memory used will be larger


The model split [download address](https://huggingface.co/wnma3mz/llama_fs_7B/tree/main)

=====================================================

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
