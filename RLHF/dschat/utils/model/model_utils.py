# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from dschat.utils.model.reward_model import RewardModel
from dschat.utils.utils import load_state_dict_into_model, print_rank_0


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in (
            "dropout",
            "attention_dropout",
            "hidden_dropout",
            "activation_dropout",
        ):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def causal_lm_model_to_fp32_loss(model):
    """Convert CausalLM model to calculate loss in fp32"""

    def causal_lm_forward(
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ):
        kwargs = (
            dict() if model.config.model_type == "llama" else dict(head_mask=head_mask)
        )
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )

        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss,) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = causal_lm_forward


def create_hf_model(
    model_class,
    model_name_or_path,
    tokenizer,
    ds_config=None,
    rlhf_training=False,
    dropout=None,
    flash_attn=False,
    torch_dtype=None,
):
    if flash_attn and torch_dtype is None:
        torch_dtype = torch.bfloat16 
        print_rank_0("You are using flash attention with torch_dtype = None. I set it to torch.bfloat16.")

    is_transformer_2 = transformers.__version__.startswith("4.36")
    if is_transformer_2:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            # use_flash_attention_2=flash_attn,
            attn_implementation="flash_attention_2" if flash_attn else "eager",
        )
    else:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            use_flash_attention_2=flash_attn,
            # attn_implementation="flash_attention_2" if flash_attn else "eager",
        )

    configure_dropout(model_config, dropout)
    if hasattr(model_config, "sliding_window"):
        if model_config.sliding_window is None:
            model_config.sliding_window = 4096

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        if flash_attn and not is_transformer_2:
            model_config._flash_attn_2_enabled = True
        model = model_class.from_config(model_config)
    else:
        if is_transformer_2:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                # use_flash_attention_2=flash_attn,
                attn_implementation="flash_attention_2" if flash_attn else "eager",
                torch_dtype=torch_dtype
            )
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                use_flash_attention_2=flash_attn,
                # attn_implementation="flash_attention_2" if flash_attn else "eager",
                torch_dtype=torch_dtype
            )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8

    return model


def create_critic_model(
    model_name_or_path,
    tokenizer,
    ds_config,
    num_padding_at_beginning=0,
    rlhf_training=False,
    dropout=None,
    zero_stage=0,
    compute_fp32_loss=False,
    flash_attn=False,
    torch_dtype=None
):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    if flash_attn and torch_dtype is None:
        torch_dtype = torch.bfloat16 
        print_rank_0("You are using flash attention with torch_dtype = None. I set it to torch.bfloat16.")


    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        dropout,
        flash_attn=flash_attn,
        torch_dtype=torch_dtype
    )
    end = time.time()
    print_rank_0(f">Creating model from_config took {end - start} seconds", None)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss,
    )

    if rlhf_training:
        # load critic model from checkpoint
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        print_rank_0(f">Loading state dict took {end - start} seconds", None)

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()

        print_rank_0(
            f">Creating model from state dict took {end - start} seconds", None
        )

    return critic_model
