import copy
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import *

import torch
import transformers
from torch.utils.data import Dataset
import pickle
from llama import Tokenizer

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
CUTOFF_LEN = 512
bos_token_id, pad_token_id, eos_token_id = 0, 0, 0


def _tokenize_fn(text: str, tokenizer: Tokenizer) -> Dict:
    """Tokenize a list of strings."""
    token_ids = tokenizer.encode(text, bos=True, eos=False)
    token_ids[0] = bos_token_id
    if CUTOFF_LEN < len(token_ids):
        token_ids = token_ids[:CUTOFF_LEN]
    else:
        token_ids += [eos_token_id] * (CUTOFF_LEN - len(token_ids))

    input_ = torch.tensor(token_ids)

    label_ = input_.clone()
    label_[label_ == pad_token_id] = IGNORE_INDEX

    return input_, input_


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids, labels = [], []
    for s, t in zip(sources, targets):
        input_, label_ = _tokenize_fn(s + t, tokenizer)
        input_ids.append(input_)
        labels.append(label_)
    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        pad_token_id = IGNORE_INDEX
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: Tokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = jload(data_path)

        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}" for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedTokenDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str):
        super(SupervisedTokenDataset, self).__init__()
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
        self.input_ids, self.labels = data_dict["input_ids"], data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


if __name__ == "__main__":
    tokenizer = Tokenizer(model_path="ckpts/tokenizer.model")

    s1 = time.time()
    fname = "datasets/alpaca_data.json"
    data = {
        "instruction": "Propose a strategy to build an effective landing page.",
        "input": "",
        "output": "A strategy to build an effective landing page is to design a page that is easy to scan, visually appealing, and speaks directly to the target audience. The page should have a clear headline that informs users of the page\u2019s purpose. It should also contain helpful, relevant information and a compelling call to action. Additionally, it should leverage A/B testing to test different versions of the page to assess its effectiveness.",
    }
    s = PROMPT_DICT["prompt_no_input"].format_map(data)
    t = data["output"]
    examples = s + t

    data_module = SupervisedDataset(fname, tokenizer)
    print("Cost Time: {}s".format(time.time() - s1))

    with open(fname.split(".json")[0] + "_token.pkl", "wb") as f:
        pickle.dump(
            {"input_ids": data_module.input_ids, "labels": data_module.labels}, f
        )

    s1 = time.time()
    with open(fname.split(".json")[0] + "_token.pkl", "rb") as f:
        data_dict = pickle.load(f)
    print("Cost Time: {}s".format(time.time() - s1))
