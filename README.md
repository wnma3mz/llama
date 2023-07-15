# LLaMA

[中文](README.md) [English](README.EN.md) [Origin](README.LLaMA.md)

本项目基于[LLaMA](https://github.com/facebookresearch/llama)修改而来。出于简单和学习的目的，本项目坚持使用一些易安装、易使用的第三方库（即不使用Transformer、PyTorch Lightning）。

完成功能：

- [X] 拆分模型，可以在多显卡小显存机器上并行推理
- [X] 完成并行微调

下一步：

- [ ] 将微调后的权重文件转换为[llama.cpp](https://github.com/ggerganov/llama.cpp)权重文件，在CPU机器上进行推理
- [ ] 对当前微调方式进行优化。但不考虑实现如Lora等相对复杂的微调方法
- [ ] 在尽可能不引入复杂依赖的前提下，加速微调且降低显存需求。手段包括但不局限于：低精度微调、GPU Offload

具体情况：

For 7B, Batch Size: 32; Seq Len: 512
单卡情况下，24G显存跑满，即一张3090

即使调整Batch Size和Seq Len，依旧需要超过12G的显存。对于小显存的显卡并不友好，因此，本项目将7B模型分别拆分为2个模型、4个模型。以便能够在小显存机器上推理。

- 2个模型的情况下，Batch Size 可以设置为8，此时每块显卡显存占用9G。[下载地址](https://huggingface.co/wnma3mz/llama_fs2_7B/tree/main)
- 4个模型的情况下，Batch Size 可以设置为32，此时每块显卡显存占用7G。[下载地址](https://huggingface.co/wnma3mz/llama_fs4_7B/tree/main)

## Setup

```
pip install -r requirements.txt
```

Then in this repository:

```
pip install -e .
```

```bash
# 当前项目中存在一个ckpts文件夹，文件架构大致如下所示。
ckpts
├── 13B
│   ├── consolidated.00.pth
│   ├── consolidated.01.pth
│   └── params.json
├── 7B
│   ├── consolidated.00.pth
│   └── params.json
└── tokenizer.model
```

## Split Model

如果是老版本的torch，需要将 `torchrun`更换为 `python3 -m torch.distributed.run`

修改split_model中的n。n为拆分后的模型数量。保存模型文件需要花费一定的时间，请耐心等待。

```bash
python3 split_model.py
```

运行结束后，在ckpts文件夹下会出现一个文件夹，7B_fs{n}。里面存放了n个模型文件和一个params.json。

```bash
ls -lh ckpts/7B_fs*/
```

## Prompt Tuning

数据集下载：[alpaca7b](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

```bash
# 需要准备的文件如下所示，其中7B_fs4是在上一步经过拆分的模型文件。而`alpaca_data`是数据集
├── ckpts
│   ├── 7B_fs4
│   │   ├── fs_consolidated.00.pth
│   │   ├── fs_consolidated.01.pth
│   │   ├── fs_consolidated.02.pth
│   │   ├── fs_consolidated.03.pth
│   │   └── params.json
│   ├── tokenizer.model
├── datasets
│   ├── alpaca_data.json
```

在 `ft_main.py`中使用函数 `train_func`进行微调始终无法得到一个好的效果。于是这里还是换成了用 `transformers.Trainer`进行训练 :(

使用 `train_func`训练3个epoch大约耗时24小时，但是 `transformers.Trainer`仅需6小时。

或许在之后有时间研究Trainer的实现细节。目前可以排除与优化器的相关因素。

```bash
# 拆分为四个模型后，在ft_main.py修改对应的配置参数
torchrun --nproc_per_node 4 ft_main.py 
```

## Inference

**Example**

微调前的推理

For 7B

```bash
# 单个模型
torchrun --nproc_per_node 1 example.py --ckpt_dir ckpts/7B --tokenizer_path ckpts/tokenizer.model
# 拆分为四个模型后
torchrun --nproc_per_node 4 example.py --ckpt_dir ckpts/7B_fs4 --tokenizer_path ckpts/tokenizer.model
# 如果拆分两个模型，则需要调整batch_size以免显存过大
torchrun --nproc_per_node 2 example.py --ckpt_dir ckpts/7B_fs2 --max_seq_len 512 --max_batch_size 5 --tokenizer_path ckpts/tokenizer.model
```

微调方式：使用HuggingFace和Peft的**Prompt Tuning**

基于[alpaca7b](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)数据集，见 `saved-alpaca7b`。

基于[simpson](https://replicate.com/blog/fine-tune-llama-to-speak-like-homer-simpson)对话数据集，见 `saved-simpsons7b`。(之后将会放出预处理后的数据集)

```bash
# 将$(ckpt_path)换成saved-alpaca7b/adapter_model.bin、saved-simpsons7b/adapter_model.bin
torchrun --nproc_per_node 4 example_ft.py --ckpt_dir ckpts/7B_fs4 --tuning_ckpt_path $(ckpt_path) --tokenizer_path ckpts/tokenizer.model
```

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

## 参考项目

- [Peft](https://github.com/huggingface/peft)
- [soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning/)
- [Prompt-Tuning](https://github.com/mkshing/Prompt-Tuning/)
