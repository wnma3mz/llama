# LLaMA 

[中文](README.md) [English](README.EN.md) [Origin](README.LLaMA.md)

This project is based on a modification of [LLaMA](https://github.com/facebookresearch/llama). For simplicity and learning purposes, this project sticks to some easy-to-install, easy-to-use third-party libraries (i.e. no Transformer, PyTorch Lightning).

For 7B, Batch Size: 32; Seq Len: 512

With a single card, 24G of video memory runs full, i.e. one 3090

Even if you adjust the Batch Size and Seq Len, you still need more than 12G of video memory. This is not friendly to small memory cards, so this project splits the 7B model into 2 models and 4 models respectively. in order to be able to reason on a small memory machine.

- In the case of 2 models, the Batch Size can be set to 8, which occupies 9G of video memory per card.[download](https://huggingface.co/wnma3mz/llama_fs2_7B/tree/main)
- For 4 models, the Batch Size can be set to 32, which is 7G per card.[download](https://huggingface.co/wnma3mz/llama_fs4_7B/tree/main)

The next step is planned, based on Prompt Tuning parallel fine tuning on four graphics cards.

Reference project

- [Peft](https://github.com/huggingface/peft)
- [soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning/)
- [Prompt-Tuning](https://github.com/mkshing/Prompt-Tuning/)


```bash
# A ckpts folder exists in the current project and the file structure is roughly as shown below.
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

## Split Model

If you have an older version of torch, you need to replace `torchrun` with `python3 -m torch.distributed.run`

Modify n in split_model. n is the number of models after splitting. It will take some time to save the model file, so please be patient.
```bash
python3 split_model.py
```

After running, a folder will appear under the ckpts folder, 7B_fs{n}. Inside it are stored n model files and a params.json.

```bash
ls -lh ckpts/7B_fs*/
```

At this point the individual model files are still 13G in size, so you can uncomment line 61 of the ``example.py`` file. After running the following command, the model file will be re-saved (which will take some time) and the size of the single model file will be 13/n G. You can continue to comment out line 61 when reasoning later to speed up the model read time.
```bash
# need to change n to the corresponding number
torchrun --nproc_per_node n example.py --ckpt_dir ckpts/7B_fsn --tokenizer_path ckpts/tokenizer.model
```

Check the saved file.
```bash
ls -lh ckpts/7B_fs*/
```

## Prompt Tuning

Not tested yet, there may be bugs.

For 7B
```bash
After splitting into four models, modify the corresponding configuration file in ft_main.py
torchrun --nproc_per_node 4 ft_main.py
```

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```bash
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

**Example**

Inference before fine-tuning

For 7B
```bash
# Single Model
torchrun --nproc_per_node 1 example.py --ckpt_dir ckpts/7B --tokenizer_path ckpts/tokenizer.model
# After splitting into four models
torchrun --nproc_per_node 4 example.py --ckpt_dir ckpts/7B_fs4 --tokenizer_path ckpts/tokenizer.model
# If you split two models, you need to adjust the batch_size to avoid oversizing the memory
torchrun --nproc_per_node 2 example.py --ckpt_dir ckpts/7B_fs2 --max_seq_len 512 --max_batch_size 5 --tokenizer_path ckpts/tokenizer.model
```

Inference After fine-tuning

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
