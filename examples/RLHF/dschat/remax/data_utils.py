# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from copy import deepcopy
from itertools import chain
from dschat.utils.data import raw_datasets
from dschat.utils.data.data_utils import (
    get_raw_dataset,
    get_shuffle_idx,
    get_raw_dataset_split_index,
)
from deepspeed.accelerator import get_accelerator

IGNORE_INDEX = -100


class PromptDataset(Dataset):
    def __init__(
        self, prompt_dataset, chosen_dataset, reject_dataset, pad_token_id
    ) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        return (
            self.prompt_dataset[idx]["input_ids"],
            self.prompt_dataset[idx]["attention_mask"],
            self.pad_token_id,
        )


def create_dataset_split(
    current_dataset,
    raw_dataset,
    tokenizer,
    end_of_conversation_token,
    max_seq_len,
    max_size=int(1e6),
):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    filtered = 0
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        prompt = raw_dataset.get_prompt(tmp_data)
        if prompt is not None:
            prompt_token = tokenizer(prompt, return_tensors="pt")
            if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                for key_word in ["input_ids", "attention_mask"]:
                    prompt_token[key_word] = prompt_token[key_word].squeeze(0).flip(0)
                prompt_dataset.append(prompt_token)
            else:
                filtered += 1
        if len(prompt_dataset) >= max_size:
            break
    print(
        f"Creating dataset {raw_dataset.dataset_name_clean} "
        f"for RL size={len(prompt_dataset)} {filtered=}"
    )

    return PromptDataset(
        prompt_dataset,
        chosen_dataset,
        reject_dataset,
        tokenizer.pad_token_id,
    )


def create_dataset(
    local_rank,
    dataset_name,
    data_split,
    output_path,
    train_phase,
    seed,
    tokenizer,
    end_of_conversation_token,
    max_seq_len,
    rebuild,
    template,
    max_size,
):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    if template == "antropic":
        print("Template = {}. Pass".format(template))
        pass
    elif template == "none":
        print("Template = {}. Pass".format(template))
        pass
    else:
        raw_dataset = raw_datasets.DatasetFormatter(raw_dataset, template)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(
        local_rank,
        output_path,
        raw_dataset.dataset_name_clean,
        seed,
        "train",
        data_split,
        1,  # train_phase - 1,
        len(train_dataset),
        rebuild,
    )
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(
        train_dataset,
        raw_dataset,
        # train_phase,
        tokenizer,
        end_of_conversation_token,
        max_seq_len,
        max_size=max_size,
    )

    # eval dataset
    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(
        local_rank,
        output_path,
        raw_dataset.dataset_name_clean,
        seed,
        "eval",
        data_split,
        1,  # train_phase - 1,
        len(eval_dataset),
        rebuild,
    )
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(
        eval_dataset,
        raw_dataset,
        # train_phase,
        tokenizer,
        end_of_conversation_token,
        max_seq_len,
    )
    return train_dataset, eval_dataset


def create_rl_prompt_dataset(
    local_rank,
    data_path,
    data_split,
    output_path,
    train_phase,
    seed,
    tokenizer,
    max_seq_len,
    end_of_conversation_token="<|endoftext|>",
    sft_only_data_path=[],
    reload=False,
    template="none",
    max_size=int(1e6),
):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}_template{template}_maxsize{max_size}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(
        fname.encode()
    ).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name()
    )
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f"Creating prompt dataset {data_path}, {reload=}")
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank,
                data_path[0],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload,
                template=template,
                max_size=max_size,
            )
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank,
                    d_path,
                    data_split,
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload,
                    template=template,
                    max_size=max_size,
                )
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)
