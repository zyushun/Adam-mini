import math
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import math
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adam_mini(torch.optim.Optimizer):
    def __init__(
            self,
            named_parameters: Iterable[Tuple[str, nn.Parameter]],
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            *,
            model_sharding: bool = None,
            dim: int = 2048,
            n_heads: int = 32,
            n_kv_heads: Optional[int] = None,
    ):

        '''
        This is the official implementation of Adam-mini (version 1.0.3).

        Paper: [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793).

        Github repo: https://github.com/zyushun/Adam-mini

        Arguments:
            named_parameters ('Iterable[Tuple[str, nn.Parameter]]'): Iterable of named parameters to optimize or dictionaries defining parameter groups. Usually set to model.named_parameters()

            lr (`float`, *optional*, defaults to 0.001): The learning rate to use.

            betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`): Same as Adam's betas parameters (b1, b2).

            eps (`float`, *optional*, defaults to 1e-06): Same as Adam's epsilon for numerical stability.

            weight_decay (`float`, *optional*, defaults to 0.0): Decoupled weight decay to apply.

            model_sharding (`bool`, *optional*, defaults to None): Set to True if you are using model parallelism with more than 1 GPU, including FSDP and zero_1,2,3 in Deepspeed. Set to False if otherwise. Due to the historical reason, this argument is deprecated since version 1.0.2. We will assume that model parallelism is always used. We will remove this argument in the future version.

            dim (`int`, *optional*, defaults to 2048): Dimension for hidden features. Can be left unspecified if training non-transformer models.

            n_heads (`int`, *optional*, defaults to 32): Number of attention heads. Can be left unspecified if training non-transformer models.

            n_kv_heads (`int`, *optional*, defaults to None): Number of heads for Key and Value. Or equivalently, number of query groups in Group Query Attention. Also known as "n_query_groups". If not specified, it will be equal to n_head. Can be left unspecified if training non-transformer models.

        Example:

        ```python
        optimizer = Adam_mini(
                    named_parameters = model.named_parameters(),
                    lr = lr,
                    betas = (beta1,beta2),
                    eps = eps,
                    weight_decay = weight_decay,
                    dim = model_config.dim,
                    n_heads = model_config.n_heads,
                    n_kv_heads = model_config.n_kv_heads,
                    )
        ```

        '''

        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        self.world_size = torch.cuda.device_count()

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.dim == int(self.dim):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not self.n_heads == int(self.n_heads):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        if model_sharding is not None:
            print(
                "UserWarning: model_sharding is deprecated since version 1.0.2. This argument is always set True. We will remove this argument in the future version.")

        # Embedding layer. Use AdamW updates for this block
        self.embd_names = {"embed", "embd", "wte"}
        # Output layers. Use AdamW updates for this block
        self.output_names = {"lm_head.weight", "output.weight"}
        # Query and Keys, will  assign lrs by heads
        self.wqk_names = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}
        # MLPs
        # self.mlp_names = {"feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight"}
        self.mlp_names = {}  # Default empty. We will use one learning rate for one layer in MLP block. If not empty, we will use per-neuron vmean for MLPs. Our ablation study shows that using one lr is the same as using per-neuron lr for MLPs.

        optim_groups = []
        count_embd = count_output = count_wqk = 0
        for param_name, param in named_parameters:
            if not param.requires_grad:
                continue
            print('Adam-mini found the param block with name:', param_name)
            state = {}
            state["name"] = param_name
            state["params"] = param
            if "norm" in param_name or "ln_f" in param_name:
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay
            if any(embd_name in param_name for embd_name in self.embd_names):
                count_embd += 1
            if any(output_name in param_name for output_name in self.output_names):
                count_output += 1
            if any(wqk_name in param_name for wqk_name in self.wqk_names):
                count_wqk += 1
                assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                state["head_numel"] = self.dim * self.dim // self.n_heads

            if any(mlp_name in param_name for mlp_name in self.mlp_names):
                state["neuron_numel"] = self.dim

            optim_groups.append(state)

        print(
            f'Adam-mini found {count_embd} embedding layers, {count_output} output layers, {count_wqk} Querys and Keys.')

        if count_embd == 0:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the name of your embedding layer'). ")
        if count_output == 0:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found. If you are training Transformers (without weight-tying), please check the name of your output layer and manually add them to 'self.output_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.output_names.add('the name of your output layer').  Please ignore this warning if you are using weight-tying.")
        if count_wqk == 0:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found. If you are training Transformers, please check the name of your Query and Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding two additional lines of code: optimizer.wqk_names.add('the name of your Query' ); optimizer.wqk_names.add('the name of your Key'). ")

        if count_output + count_embd + count_wqk == 0:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers.")

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:

                state = self.state[p]
                if any(embd_name in name for embd_name in self.embd_names) or any(output_name in name for output_name in
                                                                                  self.output_names):  # this is for embedding and output layer
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    grad = p.grad  # .to(torch.float32)
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)

                elif any(wqk_name in name for wqk_name in self.wqk_names):  # this is for query and key
                    if p.grad is None:
                        continue
                    head_numel = group["head_numel"]
                    if len(state) == 0:
                        m = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["m"] = m.view(-1, head_numel)
                        state["head_per_gpu"] = state["m"].size(0)  # this is head per gpu
                        state["step"] = 0
                        # NOTE: We must use `zeros_like` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(state["head"])
                        state["vmean"] = torch.zeros_like(state["m"][0:state["head_per_gpu"], 0:1],
                                                          memory_format=torch.preserve_format)

                    grad = p.grad  # .to(torch.float32)
                    head_per_gpu = state["head_per_gpu"]
                    grad = grad.view(head_per_gpu, head_numel)
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(head_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)

                elif any(mlp_name in name for mlp_name in
                         self.mlp_names):  # MLP blocks. If True, then Adam-mini will use one lr per output neuron. This will avoid all_reduce when the single MLP block is sharded on different GPUs. However, we find this design will not boost performance. By default, we will not enter here
                    if p.grad is None:
                        continue
                    neuron_numel = group["neuron_numel"]
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format).view(-1, neuron_numel)
                        state["step"] = 0
                        state["neuron_per_gpu"] = state["m"].size(0)  # this is neuron per gpu
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(1, device=p.device)
                        # state["vmean"] = p.new_zeros(1)
                        state["vmean"] = torch.zeros_like(state["m"][0:state["neuron_per_gpu"], 0:1],
                                                          memory_format=torch.preserve_format)

                    grad = p.grad  # .to(torch.float32)
                    neuron_per_gpu = state["neuron_per_gpu"]
                    grad = grad.view(neuron_per_gpu, neuron_numel)
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(neuron_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)

                else:  # other blocks. By default, this is for values, projections, MLPs, and LayerNorms.
                    if len(state) == 0:
                        block_numel = torch.tensor(p.numel()).to(torch.float32).to(device)
                        reduced = False
                        if (self.world_size > 1):
                            tensor_list = [torch.zeros_like(block_numel) for _ in range(self.world_size)]

                            dist.all_gather(tensor_list, block_numel)
                            s = 0
                            block_numel = 0
                            for d in tensor_list:
                                if (d > 0):
                                    s = s + 1
                                block_numel = block_numel + d
                            if (s >= 2):
                                reduced = True

                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["reduced"] = reduced
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(1, device=p.device)
                        # state["vmean"] = p.new_zeros(1)
                        state["vmean"] = torch.zeros_like(torch.sum(p * p), memory_format=torch.preserve_format)
                        state["block_numel"] = block_numel.item()
                    if p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(p * p))
                    else:
                        grad = p.grad  # .to(torch.float32)
                        tmp_lr = torch.sum(grad * grad)

                    if (state["reduced"]):
                        if "device_mesh" in dir(tmp_lr):
                            # when tmp_lr is a  DTensor in TorchTitan
                            lr_local = tmp_lr.to_local()
                            dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                            tmp_lr.redistribute(placements=[Replicate()])
                        else:
                            # when tmp_lr is a  standard tensor
                            # print(f"...... dist all reduce.......")
                            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                    if (p.grad is None):
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * (stepsize.to(state["m"].device))
                    update.mul_(lr)
                    p.add_(-update)

        return loss


