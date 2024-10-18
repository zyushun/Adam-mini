# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16),
    "20M": ModelArgs(dim=256, n_layers=4, n_heads=4),
    "22M": ModelArgs(dim=256, n_layers=6, n_heads=4),
    "35M": ModelArgs(dim=384, n_layers=6, n_heads=6),
    "39M": ModelArgs(dim=384, n_layers=8, n_heads=6),
    "60M": ModelArgs(dim=512, n_layers=8, n_heads=8),
    "67M": ModelArgs(dim=512, n_layers=10, n_heads=8),
    "102M": ModelArgs(dim=640, n_layers=12, n_heads=10),
    "134M": ModelArgs(dim=768, n_layers=12, n_heads=12),
    "162M": ModelArgs(dim=768, n_layers=16, n_heads=12),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=16),
    "297M": ModelArgs(dim=1024, n_layers=18, n_heads=16),
    "360M": ModelArgs(dim=1152, n_layers=18, n_heads=16),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}

llama3_configs = {
    "debugmodel": ModelArgs(dim=1024, n_layers=16, n_heads=16, rope_theta=500000),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
}
