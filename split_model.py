# coding: utf-8
import torch
from tabulate import tabulate
import os

# Split the 7B model to n files for parallel


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
    n = 2
    base_ckpt = torch.load(
        os.path.join(ckpt_dir, "consolidated.00.pth"), map_location="cpu"
    )

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
            ckpt[k] = w.clone()

    new_ckpt_dir = ckpt_dir + f"_fs{n}"
    if not os.path.isdir(new_ckpt_dir):
        os.makedirs(new_ckpt_dir)

    params_f = os.path.join(ckpt_dir, "params.json")
    os.system(f"cp {params_f} {new_ckpt_dir}")
    # It has a few problems. It's still 13GB after each file is stored. A less elegant approach is to reload the model and then save it, so that you get four 13/n G files.
    for i, ckpt in enumerate(split_ckpt_lst):
        torch.save(
            ckpt,
            os.path.join(new_ckpt_dir, f"fs_consolidated.{str(i).zfill(2)}.pth"),
        )
