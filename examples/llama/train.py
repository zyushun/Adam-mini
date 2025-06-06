# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os
import time

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel

from torchtitan.checkpoint import CheckpointManager
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, create_tokenizer
from torchtitan.float8_linear import build_fp8_linear
from torchtitan.logging_utils import init_logger, logger
from torchtitan.lr_scheduling import get_lr_schedulers
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)
from torchtitan.parallelisms.pipelining_utils import build_pipeline_schedule
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.utils import (
    Color,
    dist_max,
    dist_mean,
    get_metrics_rank,
    get_num_flop_per_token,
    get_num_params,
    get_peak_flops,
    init_distributed,
    NoColor,
    set_pg_timeouts,
)

from adafactor import Adafactor_zhai, Adafactor
from adalomo import AdaLomo
from adam_mini import Adam_mini
import torch_optimizer as optim
import wandb


@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


def build_optimizers(model_parts, job_config: JobConfig, world_mesh=None):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model, world_mesh=None):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = False # job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            print(f"======>>>>> Using AdamW optimizer")
            logger.info(f"======>>>>> Using AdamW optimizer, lr = {lr}")
        elif name == "adam_mini":
            optimizer = Adam_mini(
                        named_parameters = model.named_parameters(),
                        lr = lr,
                        betas = (0.9,0.95),
                        weight_decay = 0.1,
                        dim = model_config.dim,
                        n_heads = model_config.n_heads,
                        n_kv_heads = model_config.n_kv_heads,
                        )
            optimizer.wv_names = {} # For experiments with relatively small total steps  (like the 8B and 13B experiments here, we only run for 10k steps), we apply a single lr for Value and find it performs a bit better. Please comment this line if your total steps is larger than 10k or 20k or more.

            logger.info(f"======>>>>> Using Adam-mini optimizer, lr = {lr}")

        elif name == "adafactor_zhai":
                
                beta1 = 0.9
                beta2 = job_config.optimizer.beta2
                eps2 = (job_config.optimizer.eps, 1e-3)
                clip_threshold = 1.0
                weight_decay = 0.1

                optimizer = Adafactor_zhai( model.parameters(),
                            lr= lr,
                            beta1 = beta1,
                            beta2 = beta2,
                            eps2= eps2,
                            clip_threshold=clip_threshold,
                            # decay_rate=-0.8,
                            weight_decay=weight_decay,
                            scale_parameter=False,
                            relative_step=True,
                            warmup_init=True)
                logger.info(f"======>>>>> Using Adafactor-Zhai-version optimizer with beta1 = {beta1}, beta2 = {beta2}, eps2 = {eps2}, clip_threshold = {clip_threshold}, weight_decay = {weight_decay}")
        elif name == "adafactor":
                beta1 = 0.9
               # beta2 = 0.95#0.999
                eps2 = (1e-30, 1e-3)
                clip_threshold = 1.0
                decay_rate=-0.8
                weight_decay = 0.1

                optimizer = Adafactor( model.parameters(),
                            lr= lr,
                            beta1 = beta1,
                            eps2= eps2,
                            clip_threshold=clip_threshold,
                            decay_rate=decay_rate,
                            weight_decay=weight_decay,
                            scale_parameter=True,
                            relative_step=True,
                            warmup_init=True)

                logger.info(f"======>>>>> Using Adafactor optimizer with beta1 = {beta1}, beta2 = {beta2}, eps2 = {eps2}, clip_threshold = {clip_threshold}, decay_rate = {decay_rate}, weight_decay = {weight_decay}")
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()
            #pass

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])



def validate(job_config, model, data_loader_val, loss_fn, current_step):
    model.eval()
    loss_list = []
    total_tokens = 0
    num_val_batch = job_config.training.num_val_batch # limit the number of data for validation
    current_batch_idx = 0
    logger.info(f"Calculating validation loss...")
    with torch.no_grad():
        for batch in data_loader_val:
            current_batch_idx  += 1
            if current_batch_idx > num_val_batch:
                break
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            pred = model(input_ids)
            loss = loss_fn(pred, labels)
            loss_list.append( loss.item() )
            total_tokens += labels.numel()

    avg_loss = np.mean(loss_list)
    logger.info(f"Validation completed: step: {current_step}, val loss: {avg_loss} val token: {total_tokens}")
    model.train()
    return avg_loss


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = Color if job_config.metrics.enable_color_printing else NoColor

    # take control of garbage collection to avoid stragglers
    _gc_freq = job_config.training.gc_freq
    gc.disable()
    gc.collect(1)

    # init distributed
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    init_distributed(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
        dp_mesh = None # is this correct?

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # validation dataloader use c4 mini

    data_loader_val = build_hf_data_loader(
        "c4_mini",
        "./torchtitan/datasets/c4_mini/",
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # loss_parallel enables dispatching to efficient loss operators
    loss_parallel_ctx = (
        loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    global model_config
    model_config = models_config[model_name][job_config.model.flavor]

    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len
    
    run_id = job_config.job.description + job_config.model.flavor + job_config.optimizer.name + str(job_config.optimizer.lr)
    logger.info(f"=========> Currently running: {run_id}  {job_config.metrics.wandb_comment}")
    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")

    
    with torch.device("meta"):
        whole_model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(whole_model, job_config)

    # log model size
    model_param_count = get_num_params(whole_model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(whole_model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset} "
        f"num_flop_per_token: {num_flop_per_token:,} "
    )


    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor()
    # obtain the peak flops of bf16 type for MFU calculation
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)

    if parallel_dims.pp_enabled:
        stages, model_parts = models_pipelining_fns[model_name](
            whole_model, world_mesh, parallel_dims, job_config, device, model_config
        )
    else:
        # In 1D/2D cases or PP with simple schedules, model_parts is just one item
        # for PP with looped schedules, each item is one stage-model-chunk
        # we iterate all model_parts for applying SPMD parallelism, compilation, optimizer, and checkpointing
        model_parts = [whole_model]

    # apply PT-D DP/TP parallelisms and activation checkpointing
    model_parts = [
        models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
        for m in model_parts
    ]

    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    for model in model_parts:
        model.to_empty(device=init_device)

    if parallel_dims.pp_enabled:
        pp_schedule = build_pipeline_schedule(
            job_config, parallel_dims, stages, loss_fn
        )
    else:
        # If PP is enabled, we can't rely on init_weights, because some layers are missing.
        # In the future, we may make init_weights handle missing layers, but also have to consider RNG seed propagation.
        # allocate sharded model on GPU and initialize weights via DTensor
        whole_model.init_weights()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config, world_mesh=dp_mesh)
    lr_schedulers = get_lr_schedulers(optimizers.optimizers, job_config)


    metric_logger = build_metric_logger(
        job_config, metrics_log_rank=get_metrics_rank(world_mesh, parallel_dims), run_id = run_id
    )

    if job_config.metrics.enable_wandb and global_rank == 0:
        # if torch.distributed.get_rank() == 0 
        logger.info("Initializing wandb")
        run_id = job_config.model.name + job_config.model.flavor + job_config.optimizer.name + str(job_config.optimizer.lr) + job_config.metrics.wandb_comment
        wandb.init(project=job_config.job.description, name= run_id)


    train_state = TrainState()

    # train loop
    for model in model_parts:
        model.train()

    # load initial checkpoint
    job_config.checkpoint.folder +=  f"/{job_config.model.name}_{job_config.model.flavor}/{job_config.optimizer.name}"
    
    checkpoint = CheckpointManager(
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        dataloader=data_loader,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load(resume = job_config.checkpoint.resume)

    if parallel_dims.pp_enabled and not checkpoint_loaded:
        raise RuntimeError(
            "Pipeline Parallelism requires meta-initialization and loading seed checkpoint. "
            "Please run `./create_seed_checkpoint.sh` and rerun training with `--checkpoint.enable_checkpoint`"
        )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    checkpoint.reset()

    # variables used to keep info for metrics logging
    losses_since_last_log: List[float] = []
    ntokens_since_last_log = 0
    ntokens_total_train = 0
    data_loading_times: List[float] = []
    time_last_log = timer()
    gpu_memory_monitor.reset_peak_stats()

    # train loop
    logger.info(f"Training starts at step {train_state.step + 1}")
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            if train_state.step > 1 and train_state.step % _gc_freq == 0:
                gc.collect(1)

            if train_state.step % job_config.training.val_interval  == 0:
                validate(job_config, model, data_loader_val, loss_fn, train_state.step)

            # # get batch
            # data_load_start = timer()
            # batch = next(data_iterator)
            # input_ids, labels = batch
            # ntokens_since_last_log += labels.numel()
            # data_loading_times.append(timer() - data_load_start)

            # input_ids = input_ids.cuda()
            # labels = labels.cuda()
            optimizers.zero_grad()

            if parallel_dims.pp_enabled:
                logger.info('are we here? pipeline parallel forward / backward inside step() call') # False
                # pipeline parallel forward / backward inside step() call
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                with loss_parallel_ctx():
                    if pp_mesh.get_local_rank() == 0:
                        pp_schedule.step(input_ids)
                    elif is_last_stage:
                        losses = []
                        pp_schedule.step(target=labels, losses=losses)
                    else:
                        pp_schedule.step()

                # accumulate losses across pipeline microbatches
                loss = (
                    torch.mean(torch.stack(losses))
                    if is_last_stage
                    else torch.Tensor([-1.0])
                )
            else:
                # Non-PP forward / backward
                # with loss_parallel_ctx():
                #     logger.info('are we here? Non-PP forward / backward') # True
                #     pred = model(input_ids)
                #     loss = loss_fn(pred, labels)
                #     # pred.shape=(bs, seq_len, vocab_size)
                #     # need to free to before bwd to avoid peaking memory
                #     del pred
                #     loss.backward()
                
                for microbatch_idx in range(job_config.training.grad_accumulation_steps):

                    # get batch
                    data_load_start = timer()
                    batch = next(data_iterator)
                    input_ids, labels = batch
                    ntokens_since_last_log += labels.numel() * world_size
                    ntokens_total_train  += labels.numel() * world_size
                    data_loading_times.append(timer() - data_load_start)
                    
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()

                    # model.set_requires_gradient_sync(microbatch_idx==(job_config.training.grad_accumulation_steps-1)) # OOM error

                    with loss_parallel_ctx():
                        # print(f" train_state.step {train_state.step}, microbatch_idx: {microbatch_idx}, ntokens_since_last_log {ntokens_since_last_log}, rank {torch.distributed.get_rank()}")

                        pred = model(input_ids)
                        loss_unnormalized = loss_fn(pred, labels)
                        del pred

                        loss = loss_unnormalized / job_config.training.grad_accumulation_steps

                        losses_since_last_log.append(loss_unnormalized) # need to log the un-normalized loss. It will be normalized later in the log function
         
                        loss.backward()

            # clip gradients
            for model in model_parts:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), job_config.training.max_norm, foreach=True
                )

            # optimizer step
            checkpoint.wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # losses_since_last_log.append(loss)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = (
                    np.mean(losses),
                    np.max(losses),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh).item(),
                        dist_max(max_loss, dp_mesh).item(),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = timer() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = np.mean(data_loading_times)
                time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                # logger.info(
                #     f"optimizer: {job_config.optimizer.name}"
                #     f"{color.cyan}step: {train_state.step:2}  "
                #     f"{color.green}loss: {global_avg_loss:7.4f}  "
                #     f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                #     f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                #     f"{color.blue}wps: {round(wps):,}  "
                #     f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                # )
                if  "adafactor" in job_config.optimizer.name:
                    # logger.info(
                    #         )
                    logger.info(f"optimizer: {job_config.optimizer.name} "
                                f"lr: {job_config.optimizer.lr} "
                                f"step: {train_state.step} "
                                f"loss: {global_avg_loss:7.4f} "
                                f"memory/max_active(GiB): {gpu_mem_stats.max_active_gib} " 
                                f"memory/max_active(%): {gpu_mem_stats.max_active_pct} "
                                f"memory/max_reserved(GiB): {gpu_mem_stats.max_reserved_gib} "
                                f"memory/max_reserved(%): {gpu_mem_stats.max_reserved_pct} "
                                f"wps: {round(wps):,}  " # this is throughput tokens per second
                                f"mfu: {mfu:.2f}%{color.reset} "
                                f"total_time since last log (s): {time_end_to_end:7.4f} "
                                f"total_token trained: {ntokens_total_train  / 1e9} B " # assume 2 gpu
                                # f"parallel_dims.model_parallel_size: {parallel_dims.model_parallel_size} " This is 1
                            )

                else:
                    logger.info(f"optimizer: {job_config.optimizer.name} ")
                    logger.info(f"step: {train_state.step} "
                                f"loss: {global_avg_loss:7.4f} "
                                f"memory/max_active(GiB): {gpu_mem_stats.max_active_gib} " 
                                f"memory/max_active(%): {gpu_mem_stats.max_active_pct} "
                                f"memory/max_reserved(GiB): {gpu_mem_stats.max_reserved_gib} "
                                f"memory/max_reserved(%): {gpu_mem_stats.max_reserved_pct} "
                                f"wps: {round(wps):,}  " # this is throughput tokens per second
                                f"mfu: {mfu:.2f}%{color.reset} "
                                f"total_time since last log (s): {time_end_to_end:7.4f} "
                                f"total_token trained: {ntokens_total_train  / 1e9} B " # assume 2 gpu
                                # f"parallel_dims.model_parallel_size: {parallel_dims.model_parallel_size} " This is 1
                            )
                    
    
                
                if global_rank == 0  and job_config.metrics.enable_wandb:
                    wandb.log({
                        "test": 1,
                        "step": train_state.step,
                        "loss": global_avg_loss,
                        "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                        "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                        "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                        "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                        "wps": wps, # this is throughput tokens per second
                        "mfu": mfu,
                        "total_time since last log (s)": time_end_to_end, 
                        "total_token trained (B)": ntokens_total_train   / 1e9,
                        },
                        step = train_state.step)
            
                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = timer()
                gpu_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            ) # save in the end?

            # signals the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()

            if memory_profiler:
                memory_profiler.step()

            # Reduce timeout after first train step for faster signal (assumes lazy init, compile are finished)
            if train_state.step == 1:
                set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    destroy_process_group()
