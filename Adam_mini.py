import torch
from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist
from torch.optim.optimizer import _dispatch_sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.distributed._tensor import Replicate


class Adam_mini(Optimizer):
    def __init__(
            self,
            model=None,
            weight_decay=0.1,
            lr=1,
            betas=(0.9, 0.999),
            eps=1e-8,
            model_sharding=False,
            n_feature=2048,
            n_head=32,
            n_kv_head=None,
    ):
        '''
        model: the model you are training.

        model_sharding: set to True if you are using model parallelism with more than 1 GPU, including FSDP and zero_1,2,3 in Deepspeed. Set to False if otherwise.

        n_feature: dimension for hidden feature.  Could be unspecified if you are training non-transformer models.

        n_head: number of attention heads. Could be unspecified if you are training non-transformer models.

        n_kv_head: number of head for Key and Value. Or equivalently, number of query groups in Group query Attention. Also known as "n_query_groups".  If not specified, it will be equal to n_head. Could be unspecified if you are training non-transformer models.
        '''

        self.n_feature = n_feature
        self.n_head = n_head
        if n_kv_head is not None:
            self.n_kv_head = n_kv_head
            assert self.n_head % self.n_kv_head == 0
        else:
            self.n_kv_head = self.n_head
        self.model = model
        self.world_size = torch.cuda.device_count()
        self.model_sharding = model_sharding
        if self.model_sharding:
            print("=====>>> Adam-mini is using model_sharding")

        optim_groups = []
        count_embd = 0
        count_output = 0
        count_qk = 0
        for name, param in self.model.named_parameters():
            print('Detected param blocks by Adam-mini:', name)
            if param.requires_grad:
                dic = {}
                dic["name"] = name
                dic["params"] = param
                if ("norm" in name or "ln_f" in name):
                    dic["weight_decay"] = 0
                else:
                    dic["weight_decay"] = weight_decay

                if ("embed_tokens" in name or "wte" in name or "tok_embeddings" in name):
                    count_embd += 1

                if ("lm_head" in name or "output.weight" in name):
                    count_output += 1

                if ("self_attn.q_proj.weight" in name or "wq.weight" in name):
                    count_qk += 1

                    dic["parameter_per_head"] = self.n_feature * self.n_feature // self.n_head
                    if (self.n_feature * self.n_feature % self.n_head) != 0:
                        raise ValueError("'n_feature * n_feature' is not a multiple of  n_head ")

                if ("self_attn.k_proj.weight" in name or "wk.weight" in name):
                    count_qk += 1

                    dic["parameter_per_head"] = self.n_feature * self.n_feature // self.n_kv_head
                    if (self.n_feature * self.n_feature % self.n_kv_head) != 0:
                        raise ValueError("'n_feature * n_feature' is not a multiple of  n_kv_head ")

                if ("attn.attn.weight" in name or "attn.qkv.weight" in name or "attn.c_attn.weight" in name):
                    count_qk += 1
                    dic["n_head"] = self.n_head
                    dic["q_per_kv"] = self.n_head // self.n_kv_head
                    if (self.n_head % self.n_kv_head) != 0:
                        raise ValueError("'n_head' is not a multiple of n_kv_head ")

                optim_groups.append(dic)

        if count_embd == 0:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_blocks' of Adam-mini.")
        if count_output == 0:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found.  If you are training Transformers, please check the name of your output layer and manually add them to 'self.embd_blocks' of Adam-mini")
        if count_qk == 0:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found.  If you are training Transformers, please check the name of your Query and Key in attention blocks and manually add them to 'self.qk_blocks' of Adam-mini")

        if count_output + count_embd + count_qk == 0:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers.")

        # embd_blocks, including embd and output layers. Use normal adamW updates for these blocks
        self.embd_blocks = {
            "embed_tokens", "embed", "embd", "wte", "lm_head", "tok_embeddings", "output.weight"
        }
        # Query and Keys, will  assign lrs by heads
        self.qk_blocks = {
            "k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight", "q.weight", "k.weight"
        }
        # fused attn blocks, will separate in to Q, K, V and assign lrs accordingly. This is used in TinyLlama
        self.fused_attn_blocks = {"attn.attn.weight", "attn.qkv.weight", "attn.c_attn.weight"}

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.n_feature == int(self.n_feature):
            raise ValueError("Invalid n_feature value: {}".format(self.n_feature))
        if not self.n_head == int(self.n_head):
            raise ValueError("Invalid n_head value: {}".format(self.n_head))
        if not self.n_kv_head == int(self.n_kv_head):
            raise ValueError("Invalid n_kv_head value: {}".format(self.n_kv_head))

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)

        super(Adam_mini, self).__init__(optim_groups, defaults)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                lr = group["lr"]
                name = group["name"]
                eps = group["eps"]

                for p in group["params"]:
                    state = self.state[p]

                    if any(block in name for block in self.embd_blocks):  # this is for embedding and output layer
                        if p.grad is None:
                            continue
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["v"] = torch.zeros_like(p.data).to(torch.float32)

                        grad = p.grad.data.to(torch.float32)

                        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                        stepsize = lr / bias_correction_1
                        p.addcdiv_(state["m"], h, value=-stepsize)

                    elif any(block in name for block in self.qk_blocks):  # this is for query and key

                        if p.grad is None:
                            continue

                        dim = group["parameter_per_head"]

                        if (len(state) == 0):
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)

                            state["m"] = state["m"].view(-1, dim)
                            state["head"] = state['m'].shape[0]
                            state["iteration"] = 0

                            # to reconcile with DTensor
                            state["vmean"] = torch.zeros_like(state["m"][0:state["head"], 0:1]).to(device)

                        grad = p.grad.data.to(torch.float32)
                        head = state['head']
                        grad = grad.view(head, dim)

                        tmp_lr = torch.mean(grad * grad, dim=1).unsqueeze(1).to(
                            device)  # unsqueeze to size([head, 1]) to match the size of vmean

                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(eps)
                        stepsize = ((1 / bias_correction_1) / h).view(head, 1)

                        update = state["m"] * (stepsize.to(state['m'].device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else:
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)

                    elif any(block in name for block in
                             self.fused_attn_blocks):  # this is for the attention implementation in TinyLlama
                        if p.grad is None:
                            continue
                        if (len(state) == 0):
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(group["n_head"], group["q_per_kv"] + 2, -1)
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(group["n_head"], group["q_per_kv"] + 2).to(device)

                        grad = p.grad.data.to(torch.float32)
                        grad = grad.view(group["n_head"], group["q_per_kv"] + 2, -1)

                        tmp_lr = torch.mean(grad * grad, dim=2).to(device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(eps)
                        stepsize = ((1 / bias_correction_1) / h).view(group["n_head"], group["q_per_kv"] + 2, 1)

                        update = state["m"] * (stepsize.to(state['m'].device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else:
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)


                    else:  # other blocks
                        if (len(state) == 0):
                            dimension = torch.tensor(p.data.numel()).to(device).to(torch.float32)
                            reduced = False
                            if (self.world_size > 1) and (self.model_sharding is True):
                                tensor_list = [torch.zeros_like(dimension) for _ in range(self.world_size)]

                                dist.all_gather(tensor_list, dimension)
                                s = 0
                                dimension = 0
                                for d in tensor_list:
                                    if (d > 0):
                                        s = s + 1
                                    dimension = dimension + d
                                if (s >= 2):
                                    reduced = True

                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["reduced"] = reduced

                            # actually this is just tensor(0.0), but this implementation is compatible with more general
                            # types of tensor such as  DTensor in Titan
                            state["vmean"] = torch.zeros_like(torch.sum(p.data * p.data)).to(device)
                            state["dimension"] = dimension.item()
                        if p.grad is None:
                            # actually this is just tensor(0.0), but this implementation is compatible with more general
                            # types of tensor such as  DTensor in Titan
                            tmp_lr = torch.zeros_like(torch.sum(p.data * p.data)).to(device)
                        else:
                            grad = p.grad.data.to(torch.float32)
                            tmp_lr = torch.sum(grad * grad).to(device)

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
                        tmp_lr = tmp_lr / (state["dimension"])
                        tmp_lr = tmp_lr.to(grad.device)

                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])
                        state["iteration"] += 1
                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                        state["vmean"] = (1 - beta2) * tmp_lr + beta2 * state["vmean"]
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)

                        stepsize = (1 / bias_correction_1) / h
                        update = state["m"] * (stepsize.to(state['m'].device))
                        update.mul_(lr)
                        p.add_(-update)
