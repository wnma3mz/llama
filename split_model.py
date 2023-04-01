# coding: utf-8
from pathlib import Path
import torch
from tabulate import tabulate
import os


def print_diff_key(base_ckpt, new_ckpt2):
    res = []
    for k in new_ckpt2.keys():
        if base_ckpt[k].shape == new_ckpt2[k].shape:
            continue
        base_row, base_col = base_ckpt[k].shape
        new_row, new_col = new_ckpt2[k].shape
        res.append([k, (base_row, base_col), (new_row, new_col)])
    print(tabulate(res))


if __name__ == "__main__":
    ckpt_dir = "ckpts/7B"
    base_ckpt = torch.load(
        os.path.join(ckpt_dir, "consolidated.00.pth"), map_location="cpu"
    )
    # split_ckpt = torch.load(os.path.join(ckpt_dir, "fs_consolidated.00.pth")), map_location="cpu")
    # print_diff_key(base_ckpt, split_ckpt)

    n = 4
    split_ckpt_lst = [{} for _ in range(n)]
    for k in base_ckpt.keys():
        new_w_lst = []
        if len(base_ckpt[k].shape) == 2:
            base_row, base_col = base_ckpt[k].shape
            if k == "tok_embeddings.weight":
                new_w_lst = torch.split(base_ckpt[k], base_col // n, dim=1)
            if k == "output.weight":
                new_w_lst = torch.split(base_ckpt[k], base_row // n, dim=0)
            for ch in "qkv13":
                if f"w{ch}" in k:
                    new_w_lst = torch.split(base_ckpt[k], base_row // n, dim=0)
                    flag = True
            for ch_k in ["wo", "w2"]:
                if ch_k in k:
                    new_w_lst = torch.split(base_ckpt[k], base_col // n, dim=1)
            if len(new_w_lst) == 0:
                new_w_lst = [base_ckpt[k]] * n

        for ckpt, w in zip(split_ckpt_lst, new_w_lst):
            ckpt[k] = w

    if not os.path.isdir(ckpt_dir + "_fs"):
        os.makedirs(ckpt_dir + "_fs")
    # It has a few problems. It's still 13GB after each file is stored. A less elegant approach is to reload the model and then save it, so that you get four 13/4 G files.
    for i, ckpt in enumerate(split_ckpt_lst):
        torch.save(
            ckpt,
            os.path.join(ckpt_dir + "_fs", f"fs_consolidated.{str(i).zfill(2)}.pth"),
        )
