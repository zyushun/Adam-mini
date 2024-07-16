#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import sys
import random
import time
import json
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from dschat.remax.remax_trainer import (
    DeepSpeedReMaxTrainer,
    DeepSpeedReMaxTrainerUnsupervised,
)
from dschat.remax.rlhf_engine import DeepSpeedRLHFEngine
from dschat.remax.perf import print_throughput_step3
from dschat.remax.data_utils import create_rl_prompt_dataset

from dschat.utils.data.data_utils import (
    create_prompt_dataset,
    MiniDataset,
    DataCollatorRLHF,
    get_unsupervised_data,
)
from dschat.utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    moving_average,
    save_zero_three_model,
    load_hf_tokenizer,
    ExponentialMovingAverage,
)
from dschat.utils.module.lora import convert_lora_to_linear_layer
from dschat.utils.io_utils import save_code, print_machine_info
from deepspeed.accelerator import get_accelerator


def parse_args():
    global writer
    parser = argparse.ArgumentParser(description="(Step 3) RLHF training arguments")

    parser.add_argument("--algo", type=str, default="remax", help="Algorithm name")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--prompt_data_source",
        type=str,
        default=None,
        choices=["reward", None],
        help="Prompt data source.",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` "
        "will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.",
    )
    parser.add_argument(
        "--max_size", type=int, default=int(1e6), help="Max size for training prompts."
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsup_coef",
        type=float,
        default=27.8,
        help="""gamma in Equation 2 from InstructGPT paper""",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam_mini"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--actor_tokenizer_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--reward_tokenizer_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )   
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader and generation purpose.",
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
        help="Mini Batch size (per device) for the training dataloader and training purpose.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Mini Batch size (per device) for the evaluation dataloader and evaluation purpose.",
    )
    parser.add_argument(
        "--generation_batches",
        type=int,
        default=1,
        help="Generate x batches to go to training mode.",
    )
    parser.add_argument(
        "--max_prompt_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_answer_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--actor_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="A seed for reproducible data processing.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action="store_true",
        help="Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed.",
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action="store_true",
        help="Unpin actor's parameters during generation. This makes generation slower but requires less memory.",
    )
    parser.add_argument(
        "--release_inference_cache",
        action="store_true",
        help="Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size.",
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help="Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature.",
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help="Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.",
    )
    parser.add_argument(
        "--flash_attn", action="store_true", help="Enable flash attention."
    )
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
        "--offload_reference_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reference model",
    )
    parser.add_argument(
        "--offload_reward_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reward model",
    )
    parser.add_argument(
        "--actor_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model.",
    )
    parser.add_argument(
        "--reference_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Reference model.",
    )
    parser.add_argument(
        "--reward_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for reward model.",
    )
    parser.add_argument(
        "--actor_gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model.",
    )
    parser.add_argument(
        "--reward_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the reward model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temeperature for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for sampling.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="kl_onestep",
        choices=["kl", "kl_onestep", "entropy", "entropy_onestep"],
        help="Penalty type.",
    )
    parser.add_argument(
        "--kl_ctl", type=float, default=0.1, help="KL penalty coefficient."
    )
    parser.add_argument(
        "--kl_with_baseline", action="store_true", help="KL with baseline value"
    )
    parser.add_argument(
        "--clip_reward_value", type=float, default=5.0, help="Reward clip coefficient."
    )
    parser.add_argument(
        "--clip_kl_value", type=float, default=0.0, help="KL clip coefficient."
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--actor_lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--actor_lora_module_name",
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
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial actor LoRA learning rate (after the potential warmup period) to use.",
    )
    ## Make EMA as an optional feature
    parser.add_argument(
        "--enable_ema", action="store_true", help="Enable EMA checkpoint for the model."
    )
    ## Mixed Precision ZeRO++
    parser.add_argument(
        "--enable_mixed_precision_lora",
        action="store_true",
        help="Enable Mixed Precision ZeRO++ for training and generation.",
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32."
        "This applies for actor model.",
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step3_tensorboard")
    ## Print actor model answers during training
    parser.add_argument(
        "--print_answers",
        action="store_true",
        help="Print prompt and answers during training",
    )
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=20,
        help="If --print_answers enabled, controls the printing interval.",
    )
    parser.add_argument(
        "--save_answers",
        action="store_true",
        help="Save prompt and answers during training",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Wether to save model checkpoint.",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=1000, help="Maximum evaluation samples"
    )
    ## template
    parser.add_argument(
        "--actor_template",
        type=str,
        default="none",
        help="Prompt template for Actor model.",
    )
    parser.add_argument(
        "--reward_template",
        type=str,
        default="none",
        help="Prompt template for reward model.",
    )
    parser.add_argument("--eval_interval", type=int, default=10, help="Eval interval")
    ## Testing
    parser.add_argument(
        "--enable_test_mode",
        action="store_true",
        help="Enable a testing mode that terminates training based on args.test_stop_step",
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help="Training non-overflow step at which to terminate training during testing.",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if (
        args.actor_zero_stage == 2
        and args.enable_hybrid_engine
        and args.offload
        and args.actor_lora_dim == 0
    ):
        pass
        # raise ValueError(
        #     "The combination of [actor_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        # )

    return args


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = (
        args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    )
    if args.prompt_data_source == "reward":
        prompt_train_dataset, prompt_eval_dataset = create_rl_prompt_dataset(
            args.local_rank,
            args.data_path,
            args.data_split,
            args.data_output_path,
            train_phase,
            args.data_seed,
            tokenizer,
            args.max_prompt_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            reload=True,
            template=args.actor_template,
            max_size=args.max_size,
        )
    else:
        prompt_train_dataset, prompt_eval_dataset = create_prompt_dataset(
            args.local_rank,
            args.data_path,
            args.data_split,
            args.data_output_path,
            train_phase,
            args.data_seed,
            tokenizer,
            args.max_prompt_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            reload=True,
            template=args.actor_template,
            max_size=args.max_size,
        )
    _, ppl_eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        1,
        args.data_seed,
        tokenizer,
        args.max_prompt_seq_len + args.max_answer_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        reload=True,
        template=args.actor_template,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len, args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        prompt_eval_sampler = RandomSampler(prompt_eval_dataset)
        ppl_eval_sampler = RandomSampler(ppl_eval_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset, seed=args.seed)
        prompt_eval_sampler = DistributedSampler(prompt_eval_dataset, seed=args.seed)
        ppl_eval_sampler = DistributedSampler(ppl_eval_dataset, seed=args.seed)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset, seed=args.seed
            )
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size,
    )

    prompt_eval_dataloader = DataLoader(
        prompt_eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=args.per_device_generation_batch_size,
    )

    ppl_eval_dataloader = DataLoader(
        ppl_eval_dataset,
        collate_fn=default_data_collator,
        sampler=ppl_eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size,
        )
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader
        )  # basically a dummy dataloader

    num_update_steps_per_epoch = (
        min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))
        * (args.per_device_generation_batch_size / args.per_device_training_batch_size)
        / args.gradient_accumulation_steps
    )
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return (
        prompt_train_dataloader,
        prompt_eval_dataloader,
        ppl_eval_dataloader,
        unsupervised_train_dataloader,
        num_total_iters,
    )


def evaluation_ppl(trainer, eval_dataloader, device, max_eval_samples=1000):
    losses = 0
    num_samples = 0
    trainer.eval()
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = trainer.actor_model(**batch, use_cache=False)

        loss = outputs.loss
        losses += loss.float()

        num_samples += len(batch["input_ids"]) * torch.distributed.get_world_size()
        if num_samples >= max_eval_samples:
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


def evaluation_reward(
    trainer,
    eval_dataloader,
    device,
    args,
    global_step,
    deterministic=False,
    print_answers=False,
    max_eval_samples=1000,
):
    eval_reward = []
    eval_length = []
    eval_kl = []
    eval_entropy = []
    num_samples = 0
    for step, batch_prompt in enumerate(eval_dataloader):
        batch_prompt = to_device(batch_prompt, device)

        exp = trainer.generate_experience(
            batch_prompt["prompt"],
            batch_prompt["prompt_att_mask"],
            step,
            deterministic=deterministic,
            print_answers=print_answers and step % args.print_answers_interval == 0,
            eval_mode=True,
        )
        reward = exp["rewards"].mean()

        prompt_length = trainer.prompt_length
        start = prompt_length - 1
        action_mask = exp["attention_mask"][:, 1:]
        answer_length = action_mask[:, start:].sum(dim=-1).float().mean()

        kl = (
            torch.sum(exp["full_kl"][:, start:-1] * action_mask[:, start:])
            / action_mask[:, start:].sum()
        )
        entropy = (
            torch.sum(exp["entropy"][:, start:-1] * action_mask[:, start:])
            / action_mask[:, start:].sum()
        )

        eval_reward.append(reward.item())
        eval_length.append(answer_length.item())
        eval_kl.append(kl.item())
        eval_entropy.append(entropy.item())

        # save eval result
        if args.save_answers and num_samples <= 100:
            assert global_step is not None and args.output_dir is not None
            save_dir = os.path.join(args.output_dir, "evaluation")
            os.makedirs(save_dir, exist_ok=True)

            prompts = trainer.tokenizer.batch_decode(
                exp["input_ids"][:, :prompt_length], skip_special_tokens=True
            )
            answers = trainer.tokenizer.batch_decode(
                exp["input_ids"][:, prompt_length:], skip_special_tokens=True
            )
            rewards = [rew.item() for rew in exp["rewards"]]

            file_path = os.path.join(save_dir, f"rank_{args.local_rank}.json")
            save_prompts_and_answers(prompts, answers, rewards, global_step, file_path)

        num_samples += len(exp["rewards"]) * torch.distributed.get_world_size()
        if num_samples >= max_eval_samples:
            break

    return (
        np.mean(eval_reward),
        np.mean(eval_length).astype(int),
        np.mean(eval_kl),
        np.mean(eval_entropy),
    )


def save_prompts_and_answers(prompts, answers, rewards, global_step, file_path):
    assert len(prompts) == len(answers), "Mismatched lengths!"
    assert file_path.endswith(".json")
    data = [
        {
            "id": i,
            "global_step": global_step,
            "prompt": prompts[i],
            "answer": answers[i],
            "reward": rewards[i],
        }
        for i in range(len(prompts))
    ]
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Determine the next id value
        next_id = data[-1]["id"] + 1 if data else 0

        # Create new entries and append them to the data list
        new_entries = [
            {
                "id": next_id + i,
                "global_step": global_step,
                "prompt": prompts[i],
                "answer": answers[i],
                "reward": rewards[i],
            }
            for i in range(len(prompts))
        ]
        data.extend(new_entries)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)


def save_model(args, rlhf_engine, tokenizer, epoch=-1):
    if args.output_dir is not None and args.save_model:
        print_rank_0("saving model ...")
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(
                rlhf_engine.actor,
                tokenizer,
                args,
                # sub_folder=f"actor_{epoch}"
            )
            if args.enable_ema:
                save_hf_format(
                    rlhf_engine.actor_ema, tokenizer, args, sub_folder="actor_ema"
                )

        if args.actor_zero_stage == 3:
            save_zero_three_model(
                rlhf_engine.actor,
                global_rank=args.global_rank,
                # save_dir=os.path.join(args.output_dir, f"actor_{epoch}"),
                save_dir=args.output_dir,
                zero_stage=args.actor_zero_stage,
            )
            if args.enable_ema:
                save_zero_three_model(
                    rlhf_engine.actor_ema,
                    global_rank=args.global_rank,
                    save_dir=os.path.join(args.output_dir, f"actor_ema_{epoch}"),
                    zero_stage=args.actor_zero_stage,
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
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = (
        args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    )
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

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

        print(f"Tensorboard logs going to: {args.tensorboard_path}")
        writer = SummaryWriter(f"{args.tensorboard_path}")

    # load tokenizer
    tokenizer = load_hf_tokenizer(
        args.actor_tokenizer_path,
        fast_tokenizer=True,
        add_special_tokens=None,
    )
    print_rank_0(
        f"[Actor Tokenizer] BOS token: {tokenizer.bos_token} EOS token: {tokenizer.eos_token} PAD token: {tokenizer.pad_token}",
        args.global_rank,
    )
    reward_tokenizer = load_hf_tokenizer(
        args.reward_tokenizer_path,
        fast_tokenizer=True,
        add_special_tokens=None,
    )
    print_rank_0(
        f"[Reward Tokenizer] BOS token: {tokenizer.bos_token} EOS token: {tokenizer.eos_token} PAD token: {tokenizer.pad_token}",
        args.global_rank,
    )
    assert (
        tokenizer.pad_token is not None and tokenizer.pad_token != tokenizer.eos_token
    )

    # load dataset
    (
        prompt_train_dataloader,
        prompt_eval_dataloader,
        ppl_eval_dataloader,
        unsupervised_train_dataloader,
        num_total_iters,
    ) = create_datasets(args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        reward_model_name_or_path=args.reward_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args,
    )

    # Mixed Precision ZeRO++
    if args.enable_mixed_precision_lora:
        assert (
            args.actor_lora_dim > 0
        ), "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    remax_trainer = (
        DeepSpeedReMaxTrainerUnsupervised
        if unsupervised_training_enabled
        else DeepSpeedReMaxTrainer
    )
    trainer = remax_trainer(rlhf_engine, args, reward_tokenizer)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(
        args.generation_batches, args.per_device_training_batch_size
    )
    unsup_mini_dataset = MiniDataset(
        args.generation_batches, args.per_device_training_batch_size
    )

    # Train!
    print_rank_0(
        f"***** Running training (total_iters={num_total_iters}) *****",
        args.global_rank,
    )
    global_step = 0
    print_machine_info(args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {1}/{args.num_train_epochs} *****",
        args.global_rank,
    )
    perplexity, _ = evaluation_ppl(
        trainer, ppl_eval_dataloader, device, args.eval_samples
    )
    eval_reward, eval_length, eval_kl, eval_entropy = evaluation_reward(
        trainer,
        prompt_eval_dataloader,
        device,
        args,
        global_step,
        deterministic=True,
        print_answers=True,
        max_eval_samples=args.eval_samples,
    )
    print_rank_0(
        f"eval reward: {eval_reward:.2f} | eval length: {eval_length:.0f} | eval kl: {eval_kl:.2f} | eval entropy: {eval_entropy:.2f} | eval ppl: {perplexity:.2f}",
        args.global_rank,
    )
    if args.enable_tensorboard and torch.distributed.get_rank() == 0:
        writer.add_scalar("eval/reward", eval_reward, global_step=global_step)
        writer.add_scalar("eval/length", eval_length, global_step=global_step)
        writer.add_scalar("eval/kl", eval_kl, global_step=global_step)
        writer.add_scalar("eval/entropy", eval_entropy, global_step=global_step)
        writer.add_scalar("eval/ppl", perplexity, global_step=global_step)
        writer.flush()

    print_machine_info(args.global_rank)
    non_overflow_step_count = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank,
        )
        for step, (batch_prompt, batch_unsupervised) in enumerate(
            zip(prompt_train_dataloader, unsupervised_train_dataloader)
        ):
            batch_prompt = to_device(batch_prompt, device)

            # prompts = batch_prompt['prompt']
            # length = prompts.size(-1)
            # if length > args.max_prompt_seq_len:
            #     prompts = prompts[:, length - args.max_prompt_seq_len:]
            #     raise ValueError("Prompt length is too long")

            out = trainer.generate_experience(
                batch_prompt["prompt"],
                batch_prompt["prompt_att_mask"],
                step,
                deterministic=False,
                print_answers=args.print_answers
                and global_step % args.print_answers_interval == 0,
            )

            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_generation_batch_size]
                )

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, unsup_loss_sum = 0, 0
                average_reward, average_regularization, average_return = 0, 0, 0
                average_length, average_kl, average_entropy = 0, 0, 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for i, (exp_data, unsup_data) in enumerate(
                    zip(exp_dataset, unsup_dataset)
                ):
                    actor_loss, returns, regularization = trainer.compute_loss(exp_data)

                    trainer.actor_model.backward(actor_loss / len(exp_dataset))

                    actor_loss_sum += actor_loss.mean()
                    average_reward += exp_data["rewards"].mean()
                    average_return += returns.mean()
                    average_regularization += regularization.mean()

                    prompt_length = trainer.prompt_length
                    start = prompt_length - 1
                    action_mask = exp_data["attention_mask"][:, 1:]
                    answer_length = action_mask[:, start:].sum(dim=-1).float().mean()
                    average_length += answer_length

                    full_kl = (
                        torch.sum(
                            exp_data["full_kl"][:, start:-1] * action_mask[:, start:]
                        )
                        / action_mask[:, start:].sum()
                    )
                    entropy = (
                        torch.sum(
                            exp_data["entropy"][:, start:-1] * action_mask[:, start:]
                        )
                        / action_mask[:, start:].sum()
                    )
                    average_kl += full_kl
                    average_entropy += entropy

                    if unsupervised_training_enabled:
                        unsup_loss = trainer.train_unsupervised(
                            unsup_data, args.unsup_coef
                        )
                        unsup_loss_sum += unsup_loss.mean()
                    else:
                        unsup_loss_sum += torch.zeros_like(actor_loss).mean()

                    inner_iter += 1
                    if args.enable_ema:
                        moving_average(
                            rlhf_engine.actor,
                            rlhf_engine.actor_ema,
                            zero_stage=args.actor_zero_stage,
                        )
                # perform batch_update here
                trainer.actor_model.step()

                end = time.time()
                training_time = end - training_start
                e2e_time = (
                    training_time + trainer.generate_time * args.generation_batches
                )  # it is an approximation, we did not include, e.g., rw forward time etc

                print_rank_0(
                    f"Epoch: {epoch + 1}/{args.num_train_epochs} | Step: {step}/{len(prompt_train_dataloader)} | Actor Loss: {actor_loss_sum / inner_iter:.4f} | Unsupervised Loss: {unsup_loss_sum / inner_iter:.4f}",
                    args.global_rank,
                )
                print_throughput_step3(
                    rlhf_engine.actor.module,
                    args,
                    e2e_time,
                    trainer.generate_time,
                    training_time,
                    args.global_rank,
                )

                average_reward = get_all_reduce_mean(average_reward).item() / inner_iter
                average_regularization = (
                    get_all_reduce_mean(average_regularization).item() / inner_iter
                )
                average_return = get_all_reduce_mean(average_return).item() / inner_iter
                average_length = get_all_reduce_mean(average_length).item() / inner_iter
                average_kl = get_all_reduce_mean(average_kl).item() / inner_iter
                average_entropy = (
                    get_all_reduce_mean(average_entropy).item() / inner_iter
                )
                average_actor_loss = (
                    get_all_reduce_mean(actor_loss_sum).item() / inner_iter
                )
                average_unsup_loss = (
                    get_all_reduce_mean(unsup_loss_sum).item() / inner_iter
                )
                actor_grad_norm = trainer.actor_model.get_global_grad_norm()
                print_rank_0(
                    f"Reward score: {average_reward:.2f} Regularization: {average_regularization:.2f} Return: {average_return:.2f} Length: {average_length:.0f} KL: {average_kl:.2f} Entropy: {average_entropy:.2f} Grad norm: {actor_grad_norm:.4f}",
                    args.global_rank,
                )
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank,
                )

                if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                    writer.add_scalar(
                        "train/actor_loss", average_actor_loss, global_step=global_step
                    )
                    writer.add_scalar(
                        "train/unsup_loss", average_unsup_loss, global_step=step
                    )
                    writer.add_scalar(
                        "train/reward", average_reward, global_step=global_step
                    )
                    writer.add_scalar(
                        "train/regularization",
                        average_regularization,
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "train/return", average_return, global_step=global_step
                    )
                    writer.add_scalar(
                        "train/length", average_length, global_step=global_step
                    )
                    writer.add_scalar("train/kl", average_kl, global_step=global_step)
                    writer.add_scalar(
                        "train/entropy", average_entropy, global_step=global_step
                    )
                    writer.add_scalar(
                        "train/actor_gradient_norm",
                        actor_grad_norm,
                        global_step=global_step,
                    )
                    writer.flush()

            if (global_step + 1) % (
                max(1, len(prompt_train_dataloader) // args.eval_interval)
            ) == 0:
                print_rank_0(
                    f"***** Evaluating policy, Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(prompt_train_dataloader)} *****",
                    args.global_rank,
                )
                perplexity, _ = evaluation_ppl(
                    trainer, ppl_eval_dataloader, device, args.eval_samples
                )
                eval_reward, eval_length, eval_kl, eval_entropy = evaluation_reward(
                    trainer,
                    prompt_eval_dataloader,
                    device,
                    args,
                    global_step,
                    deterministic=False,
                    print_answers=True,
                    max_eval_samples=args.eval_samples,
                )
                print_rank_0(
                    f"eval reward: {eval_reward:.2f} | eval length: {eval_length:.0f} | eval kl: {eval_kl:.2f} | eval entropy: {eval_entropy:.2f} | eval ppl: {perplexity:.2f}",
                    args.global_rank,
                )
                if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                    writer.add_scalar(
                        "eval/reward", eval_reward, global_step=global_step
                    )
                    writer.add_scalar(
                        "eval/length", eval_length, global_step=global_step
                    )
                    writer.add_scalar("eval/kl", eval_kl, global_step=global_step)
                    writer.add_scalar(
                        "eval/entropy", eval_entropy, global_step=global_step
                    )
                    writer.add_scalar("eval/ppl", perplexity, global_step=global_step)
                    writer.flush()
                print_machine_info(args.global_rank)
                save_model(args, rlhf_engine, tokenizer)

            if global_step % 10 == 0:
                print_machine_info(args.global_rank)
            global_step += 1
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            actor_overflow = trainer.get_overflow()

            if not actor_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break

        if args.enable_test_mode:
            break

    # Final
    print_rank_0(f"***** Evaluating at final *****", args.global_rank)
    perplexity, _ = evaluation_ppl(
        trainer, ppl_eval_dataloader, device, args.eval_samples
    )
    eval_reward, eval_length, eval_kl, eval_entropy = evaluation_reward(
        trainer,
        prompt_eval_dataloader,
        device,
        args,
        global_step,
        deterministic=False,
        print_answers=True,
        max_eval_samples=args.eval_samples,
    )
    print_rank_0(
        f"eval reward: {eval_reward:.2f} | eval length: {eval_length:.0f} | eval kl: {eval_kl:.2f} | eval entropy: {eval_entropy:.2f} | eval ppl: {perplexity:.2f}",
        args.global_rank,
    )
    if args.enable_tensorboard and torch.distributed.get_rank() == 0:
        writer.add_scalar("eval/reward", eval_reward, global_step=global_step)
        writer.add_scalar("eval/length", eval_length, global_step=global_step)
        writer.add_scalar("eval/kl", eval_kl, global_step=global_step)
        writer.add_scalar("eval/entropy", eval_entropy, global_step=global_step)
        writer.add_scalar("eval/ppl", perplexity, global_step=global_step)
        writer.flush()
    save_model(args, rlhf_engine, tokenizer)


if __name__ == "__main__":
    main()
