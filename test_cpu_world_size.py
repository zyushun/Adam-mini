"""Visible CUDA devices are not a process group. See issues 28 and 35."""
import torch
from torch import nn
from adam_mini import Adam_mini


def test_visible_gpus_do_not_start_a_collective():
    torch.cuda.device_count = lambda: 4
    layer = nn.Linear(4, 4)
    # "weight" matches none of the transformer name lists, so the step
    # takes the generic block path that used to all_gather.
    opt = Adam_mini(
        [("weight", layer.weight)],
        lr=1e-3,
        dim=4,
        n_heads=1,
        verbose=False,
    )
    assert opt.world_size == 1
    layer.weight.grad = torch.ones_like(layer.weight)
    before = layer.weight.detach().clone()
    opt.step()
    assert not torch.equal(layer.weight, before)


if __name__ == "__main__":
    test_visible_gpus_do_not_start_a_collective()
    print("ok")
