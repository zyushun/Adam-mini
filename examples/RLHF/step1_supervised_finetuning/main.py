#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import sys
import argparse
import math
import time
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
)
from dschat.utils.io_utils import save_code, print_machine_info
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput
from adam_mini import Adam_mini


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `6,2,2`"
        "will use 60%% of data for phase 1, 20%% for phase 2"
        "and 20%% for phase 3.",
    )
    parser.add_argument(
        "--sft_only_data_path",
        nargs="*",
        default=[],
        help="Path to the dataset for only using in SFT phase.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam_mini"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.",
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Training data type",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use.",
    )
    # Evaluation
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="If > 0, perform evaluation at this interval",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=5000, help="Maximum evaluation samples"
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32.",
    )
    parser.add_argument(
        "--flash_attn", action="store_true", help="Whether to use flash attention."
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Whether to use flash attention."
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step1_tensorboard")
    ## Print loss
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def evaluation_ppl(model, eval_dataloader, max_eval_samples, device):
    model.eval()
    losses = 0
    num_samples = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses += loss.float()

        num_samples += len(batch["input_ids"]) * torch.distributed.get_world_size()
        if step >= max_eval_samples:
            break
    losses = losses / (step + 1)
    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")
    return perplexity, losses.item()


def save_model(args, model, tokenizer):
    if args.output_dir is not None and args.save_model:
        print_rank_0("saving the final model ...", args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model, args.global_rank, args.output_dir, zero_stage=args.zero_stage
            )


def main():
    args = parse_args()
    args.tensorboard_path = args.output_dir

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="step1_model",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    if args.global_rank == 0:
        args.world_size = torch.distributed.get_world_size()
        with open(
            os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            for key, value in args.__dict__.items():
                json.dump({key: value}, f, ensure_ascii=False)
                f.write("\n")
        save_code(args.output_dir)

    # load tokenizer
    tokenizer = load_hf_tokenizer(
        args.tokenizer_path,
        fast_tokenizer=True,
        add_special_tokens=None,
    )
    print_rank_0(
        f"BOS token: {tokenizer.bos_token} EOS token: {tokenizer.eos_token} PAD token: {tokenizer.pad_token}",
        args.global_rank,
    )
    assert (
        tokenizer.pad_token is not None and tokenizer.pad_token != tokenizer.eos_token
    )

    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config,
        dropout=args.dropout,
        flash_attn=args.flash_attn,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else None,
    )

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank,
        )
        causal_lm_model_to_fp32_loss(model)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(
            model, args.lora_module_name, args.lora_dim
        )
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.data_seed,
        tokenizer,
        args.max_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        sft_only_data_path=args.sft_only_data_path,
        reload=True,
    )
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate
    )

    if args.optimizer == "adam_mini":
        optimizer = Adam_mini(model = model,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95),
                              weight_decay=args.weight_decay,
                              model_sharding=args.zero_stage != 0,
                              n_feature=4096,
                              n_head=32
                              )
        ds_config["zero_allow_untested_optimizer"] = True
        ds_config["zero_force_ds_cpu_optimizer"] = False
    else:
        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay, args.lora_learning_rate
        )
        optimizer = AdamOptimizer(
            optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
        )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {1}/{args.num_train_epochs} *****",
        args.global_rank,
    )
    perplexity, eval_loss = evaluation_ppl(
        model, eval_dataloader, args.eval_samples, device
    )
    print_rank_0(f"ppl: {perplexity:.2f}, loss: {eval_loss:.4f}", args.global_rank)
    print_machine_info(args.global_rank)
    if model.monitor.enabled and model.global_rank == 0:
        summary_events = [
            ("Test/ppl", perplexity, model.global_samples),
            ("Test/loss", eval_loss, model.global_samples),
        ]
        model.monitor.write_events(summary_events)

    gradient_norm = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()

        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)

            loss = outputs.loss
            model.backward(loss)
            model.step()

            end = time.time()

            gradient_norm_ = model.get_global_grad_norm()
            if gradient_norm_ is not None:
                gradient_norm = gradient_norm_

            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start, args.global_rank)

            if args.print_loss and torch.distributed.get_rank() == 0 and step % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.num_train_epochs} Step: {step}/{len(train_dataloader)}, Rank: {torch.distributed.get_rank()}, loss = {loss:.4f} grad norm = {gradient_norm:.4f}"
                )

            if model.monitor.enabled and torch.distributed.get_rank() == 0:
                summary_events = [
                    ("Train/loss", loss.item(), model.global_samples),
                    ("Train/gradient norm", gradient_norm, model.global_samples),
                ]
                model.monitor.write_events(summary_events)

            if (step + 1) % max(1, len(train_dataloader) // args.eval_interval) == 0:
                # Evaluate perplexity on the validation set.
                print_rank_0(
                    f"***** Evaluating perplexity, Epoch {epoch + 1}/{args.num_train_epochs} Step: {step}/{len(train_dataloader)} *****",
                    args.global_rank,
                )
                perplexity, eval_loss = evaluation_ppl(
                    model, eval_dataloader, args.eval_samples, device
                )
                print_rank_0(
                    f"eval ppl: {perplexity:.2f}, eval loss: {eval_loss:.4f}",
                    args.global_rank,
                )
                print_machine_info(args.global_rank)
                if model.monitor.enabled and torch.distributed.get_rank() == 0:
                    summary_events = [
                        ("Test/ppl", perplexity, model.global_samples),
                        ("Test/loss", eval_loss, model.global_samples),
                    ]
                    model.monitor.write_events(summary_events)
                model.train()

                # save_model(args, model, tokenizer)
        model.tput_timer.update_epoch_count()

    save_model(args, model, tokenizer)
    model.save_checkpoint(save_dir = args.output_dir)


if __name__ == "__main__":
    main()
