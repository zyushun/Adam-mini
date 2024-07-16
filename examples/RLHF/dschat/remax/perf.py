import os
import sys

import torch

from dschat.utils.perf import get_hf_configs, calculate_flops


# Enhanced version of the function above that provides calculations and printing for Step 3
def print_throughput_step3(
    actor_model, args, e2e_time, gen_exp_time, train_time, rank=0
):
    if rank <= 0:
        # Actor model passed here is a HF model.
        actor_hf_config = actor_model.config

        actor_num_layers, actor_hidden_size, actor_vocab_size = get_hf_configs(
            actor_hf_config
        )

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_answer_seq_len + args.max_prompt_seq_len
        batch_size = (
            args.per_device_generation_batch_size
            * args.generation_batches
            * gpus_per_model
            * 1
            if args.unsupervised_dataset_name is None
            else 2
        )
        samples_per_second = batch_size / e2e_time

        actor_checkpoint_activations_factor = (
            4 if args.actor_gradient_checkpointing else 3
        )
        if args.actor_lora_dim > 0:
            k = args.actor_lora_dim * 2 / actor_hidden_size
            actor_checkpoint_activations_factor -= 1 - k

        actor_model._num_params = sum(
            [
                p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
                for p in actor_model.parameters()
            ]
        )
        actor_params_in_billions = actor_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops

        actor_train_flops_per_iteration = calculate_flops(
            actor_checkpoint_activations_factor, batch_size, seq_length, actor_hf_config
        )

        total_train_flops = actor_train_flops_per_iteration + 0
        train_tflops = total_train_flops / (train_time * gpus_per_model * (10**12))

        gen_bs = args.per_device_generation_batch_size * gpus_per_model

        # Modified formula for calculating flops in the forward pass only
        gen_flops_per_iteration = (
            24 * gen_bs * seq_length * actor_num_layers * (actor_hidden_size**2)
        ) * (
            1.0
            + (seq_length / (6.0 * actor_hidden_size))
            + (actor_vocab_size / (16.0 * actor_num_layers * actor_hidden_size))
        )

        gen_tflops = gen_flops_per_iteration / (
            gen_exp_time * gpus_per_model * (10**12)
        )

        if actor_hf_config.torch_dtype == torch.float16:
            num_bytes = 2
        elif actor_hf_config.torch_dtype == torch.float32:
            num_bytes = 4
        else:
            num_bytes = -1

        pertok_lat = gen_exp_time / args.max_answer_seq_len
        gen_bw = 1 / pertok_lat * actor_model._num_params * num_bytes / 1e9

        total_flops_per_iteration = (
            total_train_flops + gen_flops_per_iteration * args.generation_batches
        )
        total_tflops = total_flops_per_iteration / (
            e2e_time * gpus_per_model * (10**12)
        )

        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time / batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat * 1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: {gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {args.max_answer_seq_len}"
        )
        print(f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}")
        actor_param_string = (
            f"{actor_params_in_billions:.3f} B"
            if actor_params_in_billions != 0
            else "NA"
        )
        print(f"Actor Model Parameters => {actor_param_string}")
