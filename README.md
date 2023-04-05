# LLaMA 

[中文](README.md) [English](README.EN.md) [Origin](README.LLaMA.md)

本项目基于[LLaMA](https://github.com/facebookresearch/llama)修改而来。出于简单和学习的目的，本项目坚持使用一些易安装、易使用的第三方库（即不使用Transformer、PyTorch Lightning）。

For 7B, Batch Size: 32; Seq Len: 512
单卡情况下，24G显存跑满，即一张3090

即使调整Batch Size和Seq Len，依旧需要超过12G的显存。对于小显存的显卡并不友好，因此，本项目将7B模型分别拆分为2个模型、4个模型。以便能够在小显存机器上推理。

- 2个模型的情况下，Batch Size 可以设置为8，此时每块显卡显存占用9G。[下载地址](https://huggingface.co/wnma3mz/llama_fs2_7B/tree/main)
- 4个模型的情况下，Batch Size 可以设置为32，此时每块显卡显存占用7G。[下载地址](https://huggingface.co/wnma3mz/llama_fs4_7B/tree/main)

下一步计划，在四张显卡上基于Prompt Tuning并行微调。

参考项目

- [Peft](https://github.com/huggingface/peft)
- [soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning/)
- [Prompt-Tuning](https://github.com/mkshing/Prompt-Tuning/)


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

如果是老版本的torch，需要将`torchrun`更换为`python3 -m torch.distributed.run`

修改split_model中的n。n为拆分后的模型数量。保存模型文件需要花费一定的时间，请耐心等待。
```bash
python3 split_model.py
```

运行结束后，在ckpts文件夹下会出现一个文件夹，7B_fs{n}。里面存放了n个模型文件和一个params.json。

```bash
ls -lh ckpts/7B_fs*/
```

此时单个模型文件大小依旧为13G，可以将`example.py`文件61行注释取消。在运行下面命令后，会重新保存模型文件（花费一定的时间），此时单个模型文件的大小会变为13/n G。之后进行推理时，可以继续注释61行，以加速模型读取时间。
```bash
# 需要将n改为对应数字
torchrun --nproc_per_node n example.py --ckpt_dir ckpts/7B_fsn --tokenizer_path ckpts/tokenizer.model
```

检查保存后的文件。
```bash
ls -lh ckpts/7B_fs*/
```

## Prompt Tuning

尚未测试，可能存在Bug。

For 7B
```bash
# 拆分为四个模型后，在ft_main.py修改对应的配置文件
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

微调后的推理

```bash
torchrun --nproc_per_node 4 example_ft.py --ckpt_dir ckpts/7B_fs4 --tuning_ckpt_dir ckpts/7B_ft4 --tokenizer_path ckpts/tokenizer.model
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
