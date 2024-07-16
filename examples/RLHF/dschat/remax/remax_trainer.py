# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import math
import deepspeed

# DeepSpeed Team
import torch
import torch.nn.functional as F
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0
from dschat.utils.data.data_utils import DataCollatorRLHF


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name()
    )
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f"{tag} {all_tensor}", rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = (
                hasattr(param, "ds_id")
                and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            )
            with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def timeit(func):
    def f_(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print_rank_0(f"Running {func} using time: {te - ts:.2f} sec", None)
        return result

    return f_


class RewardTemplateTransform:
    def __init__(
        self,
        max_token_len,
        inference_tp_size,
        actor_tokenizer,
        reward_tokenizer,
        actor_template,
        reward_template,
    ):
        self.max_token_len = max_token_len
        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.actor_template = actor_template
        self.reward_template = reward_template

        # self.data_collator = DataCollatorRLHF(max_token_len, inference_tp_size)

    # @timeit
    def apply(self, prompt_tokenized, seq_tokenized, attention_mask, prompt_length):
        if self.actor_template == self.reward_template:
            assert (
                self.actor_tokenizer.vocab_size == self.reward_tokenizer.vocab_size
                # and self.actor_tokenizer.pad_token == self.reward_tokenizer.pad_token
                and self.actor_tokenizer.eos_token == self.reward_tokenizer.eos_token
            ), f"{self.actor_tokenizer} != {self.reward_tokenizer}"
            return seq_tokenized, attention_mask, prompt_length
        else:
            prompt = self.actor_tokenizer.batch_decode(
                prompt_tokenized, skip_special_tokens=True
            )
            # this ensures that EOS token is not skipped due to skip_special_tokens
            ans = []
            for i in range(len(seq_tokenized)):
                ans_length = attention_mask[i, prompt_length:].sum()
                ans.append(
                    self.actor_tokenizer.decode(
                        seq_tokenized[i, prompt_length : prompt_length + ans_length],
                    )
                )
            if self.actor_template == "vicuna":
                assert (
                    prompt[0][:155]
                    == "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
                ), f"prompt is not in the vicuna form: {prompt}"
                if self.reward_template == "ultrarm":
                    prompt_new = prompt.copy()
                    for i in range(len(prompt_new)):
                        prompt_new[i] = prompt[i][155:]
                        prompt_new[i] = (
                            prompt_new[i]
                            .replace("USER:", "\nHuman:")
                            .replace("ASSISTANT:", "\nAssistant:")
                        )
                        prompt_new[i] = prompt_new[i][1:]  # remove the first \n
                else:
                    raise NotImplementedError(
                        f"reward template {self.reward_template} is not implemented for prompt template {self.actor_template}."
                    )
            elif self.actor_template == "mistral":
                """
                <s> [INST] What is your favourite condiment? [/INST]Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> [INST] Do you have mayonnaise recipes? [/INST]
                """
                if self.reward_template == "ultrarm":
                    prompt_new = prompt.copy()
                    for i in range(len(prompt_new)):
                        prompt_new[i] = (
                            prompt_new[i]
                            .replace(" [INST] ", "\nHuman: ")
                            .replace(" [/INST]", "\nAssistant: ")
                        )
                        prompt_new[i] = prompt_new[i][1:]  # remove the first \n
                else:
                    raise NotImplementedError(
                        f"reward template {self.reward_template} is not implemented for prompt template {self.actor_template}."
                    )
            else:
                raise NotImplementedError(
                    f"prompt template {self.actor_template} is not supported."
                )

            self.reward_tokenizer.padding_side = "left"
            prompt_outputs = self.reward_tokenizer(
                prompt_new,
                # max_length=self.max_token_len,
                # padding="max_length",
                padding="longest",
                return_tensors="pt",
            )
            prompt_length_new = prompt_outputs["input_ids"].shape[1]
            if prompt_length_new > prompt_length and self.actor_template == "vicuna":
                print(
                    f"It is strange that new prompts are longer than the original.\nPrompt new: {prompt_new}\nOriginal prompt:{prompt}"
                )

            self.reward_tokenizer.padding_side = "right"
            ans_outputs = self.reward_tokenizer(
                ans, return_tensors="pt", padding="longest"
            )
            if self.reward_tokenizer.add_bos_token:
                ans_outputs = {
                    "input_ids": ans_outputs["input_ids"][:, 1:],
                    "attention_mask": ans_outputs["attention_mask"][:, 1:],
                }

            seq_tokenized_new = torch.cat(
                [prompt_outputs["input_ids"], ans_outputs["input_ids"]], dim=-1
            ).to(prompt_tokenized.device)
            attention_mask_new = torch.cat(
                [prompt_outputs["attention_mask"], ans_outputs["attention_mask"]],
                dim=-1,
            ).to(prompt_tokenized.device)
            return seq_tokenized_new, attention_mask_new, prompt_length_new


class DeepSpeedReMaxTrainer:
    def __init__(self, rlhf_engine, args, reward_tokenizer=None):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.z3_enabled = args.actor_zero_stage == 3
        self.z3_ref_enbale = args.reference_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # Those value can be changed
        self.kl_ctl = args.kl_ctl
        self.clip_reward_value = args.clip_reward_value
        self.clip_kl_value = args.clip_kl_value
        self.kl_with_baseline = args.kl_with_baseline

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.generate_baseline = {"remax": True, "reinforce": False}[args.algo]

        self.reward_template_transform = RewardTemplateTransform(
            args.max_prompt_seq_len,
            args.inference_tp_size,
            self.tokenizer,
            reward_tokenizer or self.tokenizer,
            args.actor_template,
            args.reward_template,
        )
        # self.fast_greedy_sampling = args.fast_greedy_sampling
        # self.fast_greedy_sampling_step = args.fast_greedy_sampling_step
        self.penalty = args.penalty
        self.gamma = 1.0
        self.generate_time = 0.0

    def _generate_sequence(
        self,
        model,
        prompts,
        mask,
        step,
        print_answers=False,
        do_sample=True,
        synced_gpus=False,
        tag="model",
    ):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        kwargs = dict(
            do_sample=do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
        )

        with torch.no_grad():
            seq = model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=synced_gpus,
                min_new_tokens=2,
                **kwargs,
            )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt
        # without supervised fine tuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        # if print_answers:
        #     print(
        #         f"[{tag}, step={step}, rank={torch.distributed.get_rank()}], {self.tokenizer.batch_decode(seq, skip_special_tokens=True)[0]}"
        #     )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f"Dropping too short generated answer: {step=}: \n"
                    f"prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n"
                    f"answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}"
                )
                continue
            else:
                out_seq.append(seq[i : i + 1])

        if not out_seq:
            print(
                f"All generated results are too short for rank={self.args.local_rank} step={step}\n"
                f"-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n"
                f"-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}"
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq

    def generate_experience(
        self,
        prompts,
        mask,
        step,
        deterministic=False,
        print_answers=False,
        eval_mode=False,
    ):
        # ziniu: manully set the batch size for actor; otherwise it may throw an error
        if (
            hasattr(self.actor_model, "_total_batch_size")
            and self.actor_model._total_batch_size is None
        ):
            bsz = prompts.shape[0]
            self.actor_model._total_batch_size = (
                bsz * torch.distributed.get_world_size()
            )
        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(
            self.actor_model,
            prompts,
            mask,
            step,
            print_answers,
            synced_gpus=self.z3_enabled,
            do_sample=True if not deterministic else False,
        )
        if not eval_mode and self.generate_baseline:
            baseline_seq = self._generate_sequence(
                self.actor_model,
                prompts,
                mask,
                step,
                print_answers,
                synced_gpus=self.z3_enabled,
                do_sample=False,
                tag="greedy",
            )
        generate_end = time.time()
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        if not eval_mode and self.generate_baseline:
            baseline_attention_mask = baseline_seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            output = self.actor_model(
                seq, attention_mask=attention_mask, use_cache=False
            )
            output_ref = self.ref_model(
                seq, attention_mask=attention_mask, use_cache=False
            )
            (
                seq_for_reward,
                atten_for_reward,
                prompt_length_for_reward,
            ) = self.reward_template_transform.apply(
                prompts, seq, attention_mask, self.prompt_length
            )
            ts = time.time()
            reward_score = self.reward_model.forward_value(
                seq_for_reward, atten_for_reward, prompt_length=prompt_length_for_reward
            )["chosen_end_scores"].detach()
            # print_rank_0(f"computing reward takes {time.time() - ts:.2f} sec.", None)

            if not eval_mode and self.generate_baseline:
                (
                    base_seq_for_reward,
                    base_atten_for_reward,
                    base_prompt_lenthgh_for_reward,
                ) = self.reward_template_transform.apply(
                    prompts,
                    baseline_seq,
                    baseline_attention_mask,
                    self.prompt_length,
                )
                baseline_reward_score = self.reward_model.forward_value(
                    base_seq_for_reward,
                    base_atten_for_reward,
                    prompt_length=base_prompt_lenthgh_for_reward,
                )["chosen_end_scores"].detach()
            else:
                baseline_reward_score = torch.zeros_like(reward_score)

            values = torch.zeros_like(reward_score, device=reward_score.device)

        if print_answers:
            print(
                f"[model, step={step}, rank={torch.distributed.get_rank()}, reward={reward_score[0]:.2f}], {self.tokenizer.batch_decode(seq, skip_special_tokens=True)[0]}"
            )
            if not eval_mode and self.generate_baseline:
                print(
                    f"[greedy, step={step}, rank={torch.distributed.get_rank()}, reward={baseline_reward_score[0]:.2f}], {self.tokenizer.batch_decode(baseline_seq, skip_special_tokens=True)[0]}"
                )

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        log_softmax_values = F.log_softmax(logits, dim=-1)
        softmax_probs = torch.exp(log_softmax_values)
        entropy = -torch.sum(softmax_probs * log_softmax_values, dim=-1)

        log_softmax_values_ref = F.log_softmax(logits_ref, dim=-1)
        full_kl = torch.sum(
            softmax_probs * (log_softmax_values - log_softmax_values_ref), dim=-1
        )

        logprobs = log_softmax_values.gather(
            dim=-1, index=seq[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ref_logprobs = log_softmax_values_ref.gather(
            dim=-1, index=seq[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        self.generate_time = generate_end - generate_start

        return {
            "prompts": prompts,
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
            "value": values,
            "rewards": reward_score,
            "baseline_rewards": baseline_reward_score,
            "full_kl": full_kl,
            "entropy": entropy,
            "input_ids": seq,
            "attention_mask": attention_mask,
        }

    def compute_returns(self, prompts, kl_divergence, reward_score, action_mask):
        returns = torch.zeros_like(kl_divergence)
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)  # + 1
        reward_clip = torch.clamp(
            reward_score, -self.clip_reward_value, self.clip_reward_value
        )
        batch_size = kl_divergence.shape[0]
        for j in range(batch_size):
            cumulative_reward = reward_clip[j]
            cumulative_kl = 0
            for i in reversed(range(start, ends[j])):
                if self.penalty == "kl_onestep":
                    cumulative_kl = kl_divergence[j, i]
                elif self.penalty == "kl_fullstep":
                    cumulative_kl += kl_divergence[j, i]
                else:
                    raise ValueError(self.penalty)

                returns[j, i] += cumulative_kl + cumulative_reward
                cumulative_reward *= self.gamma
                cumulative_kl *= self.gamma
        return returns

    def compute_loss(self, inputs):
        # train the rlhf mode here
        prompts = inputs["prompts"]
        log_probs = inputs["logprobs"]
        ref_log_probs = inputs["ref_logprobs"]
        reward_score = inputs["rewards"]
        baseline_reward_score = inputs["baseline_rewards"]
        attention_mask = inputs["attention_mask"]
        seq = inputs["input_ids"]
        full_kl = inputs["full_kl"]
        entropy = inputs["entropy"]

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        with torch.no_grad():
            kl_divergence = -(log_probs - ref_log_probs)
            if self.kl_with_baseline:
                kl_divergence = kl_divergence + full_kl[:, :-1]
            if self.clip_kl_value:
                kl_divergence = torch.clamp(
                    kl_divergence, -self.clip_kl_value, self.clip_kl_value
                )
            kl_divergence = self.kl_ctl * kl_divergence

            reward_score = reward_score - baseline_reward_score
            returns = self.compute_returns(
                prompts, kl_divergence, reward_score, action_mask
            )

        # process the new outputs
        batch = {"input_ids": seq, "attention_mask": attention_mask}
        logits = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(logits[:, :-1, :], seq[:, 1:])

        actor_loss = self.actor_loss_fn(
            actor_log_prob[:, start:],
            returns[:, start:],
            action_mask[:, start:],
        )
        return_mean = torch.sum(returns[:, start:] * action_mask[:, start:]) / (
            action_mask[:, start:].sum()
        )
        kl_mean = torch.sum(kl_divergence[:, start:] * action_mask[:, start:]) / (
            action_mask[:, start:].sum()
        )

        return (
            actor_loss,
            return_mean,
            kl_mean,
        )

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        return actor_overflow

    def actor_loss_fn(self, logprobs, returns, mask):
        # policy gradient loss
        actor_loss = torch.sum(-returns * logprobs * mask) / mask.sum()
        return actor_loss

    def _validate_training_mode(self):
        assert self.actor_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()

    def eval(self):
        self.actor_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(
            f"{tag} global_actor_model_norm", actor_model_norm, self.args.local_rank
        )
        print_all_ranks(
            f"{tag} global_ref_model_norm", ref_model_norm, self.args.local_rank
        )
        print_all_ranks(
            f"{tag} global_reward_model_norm", reward_model_norm, self.args.local_rank
        )


class DeepSpeedReMaxTrainerUnsupervised(DeepSpeedReMaxTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
