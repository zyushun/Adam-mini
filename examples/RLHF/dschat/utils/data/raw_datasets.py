# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os

# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name, from_disk=False):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # if not dataset_name == "local/jsonfile":
        #     self.raw_datasets = load_dataset(dataset_name)
        if from_disk:
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == "local/jsonfile":
            self.raw_datasets = load_dataset(dataset_name)
        else:
            raise NotImplementedError

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample["prompt"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["chosen"]

    def get_rejected(self, sample):
        return " " + sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["rejected"]


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"] + "Assistant:"

    def get_chosen(self, sample):
        return sample["chosen"].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample["rejected"].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample["question"]["full_text"] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample["score_0"]) >= float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample["score_0"]) < float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample["score_0"]) >= float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample["question"]["full_text"] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample["score_0"]) < float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample["question"]["full_text"] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["history"] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample["history"] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample["history"] + " Assistant: " + response


# English dataset
class PvduySharegptalpacaoavicunaformatDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
        self.dataset_name_clean = "pvduy_sharegpt_alpaca_oa_vicuna_format"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        if sample["prompt"] is not None and len(sample["prompt"]) > 0:
            return (
                sample["prompt"]
                .replace("USER", "Human")
                .replace("ASSISTANT", "Assistant")
            )
        return None

    def get_chosen(self, sample):
        if sample["label"] is not None and len(sample["label"]) > 0:
            return " " + sample["label"]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if (
            sample["prompt"] is not None
            and sample["label"] is not None
            and len(sample["prompt"]) > 0
            and len(sample["label"]) > 0
        ):
            return (
                sample["prompt"]
                .replace("USER", "Human")
                .replace("ASSISTANT", "Assistant")
                + " "
                + sample["label"]
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


class LocalJsonFileDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset(
            "json",
            data_files={
                "train": chat_path + "/data/train.json",
                "eval": chat_path + "/data/eval.json",
            },
        )

    def get_train_data(self):
        if self.raw_datasets["train"] is not None:
            return self.raw_datasets["train"]
        return None

    def get_eval_data(self):
        if self.raw_datasets["eval"] is not None:
            return self.raw_datasets["eval"]
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample["prompt"] is not None:
            return " " + sample["prompt"]
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample["chosen"] is not None:
            return " " + sample["chosen"]
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample["rejected"] is not None:
            return " " + sample["rejected"]
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["prompt"] is not None and sample["chosen"] is not None:
            return " " + sample["prompt"] + " " + sample["chosen"]
        return None

    def get_prompt_and_rejected(self, sample):
        if sample["prompt"] is not None and sample["rejected"] is not None:
            return " " + sample["prompt"] + " " + sample["rejected"]
        return None


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample["INSTRUCTION"] is not None:
            return " Human: " + sample["INSTRUCTION"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["RESPONSE"] is not None:
            return " " + sample["RESPONSE"]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["INSTRUCTION"] is not None and sample["RESPONSE"] is not None:
            return (
                " Human: " + sample["INSTRUCTION"] + " Assistant: " + sample["RESPONSE"]
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample["query"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["positive_passages"][0]["text"]

    def get_rejected(self, sample):
        return " " + sample["negative_passages"][0]["text"]

    def get_prompt_and_chosen(self, sample):
        return (
            " Human: "
            + sample["query"]
            + " Assistant: "
            + sample["positive_passages"][0]["text"]
        )

    def get_prompt_and_rejected(self, sample):
        return (
            " Human: "
            + sample["query"]
            + " Assistant: "
            + sample["negative_passages"][0]["text"]
        )


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample["question"] is not None:
            return " Human: " + sample["question"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["human_answers"][0] is not None:
            return " " + sample["human_answers"][0]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["question"] is not None and sample["human_answers"][0] is not None:
            return (
                " Human: "
                + sample["question"]
                + " Assistant: "
                + sample["human_answers"][0]
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample["queries"]["zh_cn"] is not None:
            return " Human: " + sample["queries"]["zh_cn"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["answers"]["zh_cn"][0]["text"] is not None:
            return " " + sample["answers"]["zh_cn"][0]["text"]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if (
            sample["queries"]["zh_cn"] is not None
            and sample["answers"]["zh_cn"][0]["text"] is not None
        ):
            return (
                " Human: "
                + sample["queries"]["zh_cn"]
                + " Assistant: "
                + sample["answers"]["zh_cn"][0]["text"]
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index

        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name_clean,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset),
        )
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample["queries"]["ja"] is not None:
            return " Human: " + sample["queries"]["ja"] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample["answers"]["ja"][0]["text"] is not None:
            return " " + sample["answers"]["ja"][0]["text"]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if (
            sample["queries"]["ja"] is not None
            and sample["answers"]["ja"][0]["text"] is not None
        ):
            return (
                " Human: "
                + sample["queries"]["ja"]
                + " Assistant: "
                + sample["answers"]["ja"][0]["text"]
            )
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample["query"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["positive_passages"][0]["text"]

    def get_rejected(self, sample):
        return " " + sample["negative_passages"][0]["text"]

    def get_prompt_and_chosen(self, sample):
        return (
            " Human: "
            + sample["query"]
            + " Assistant: "
            + sample["positive_passages"][0]["text"]
        )

    def get_prompt_and_rejected(self, sample):
        if len(sample["negative_passages"]) > 0:
            return (
                " Human: "
                + sample["query"]
                + " Assistant: "
                + sample["negative_passages"][0]["text"]
            )
        return None


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["question"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["sentence"]

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["question"] + " Assistant: " + sample["sentence"]

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["questions"][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["paragraph"]

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return (
            " Human: " + sample["questions"][0] + " Assistant: " + sample["paragraph"]
        )

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# lmsys-chat-1M dataset
class LMSYSChatDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmsys/lmsys-chat-1m"
        self.dataset_name_clean = "lmsys_lmsys-chat-1m"

    def get_train_data(self):
        return self.raw_datasets["train"].shuffle(seed=42)

    def get_eval_data(self):
        print(
            "There is no eval data for {}. We set the first 2000 'train' data to 'eval'.".format(
                self.dataset_name
            )
        )
        return self.raw_datasets["train"].select(range(2000))

    def get_prompt(self, sample):
        history = sample["conversation"]
        conv = []
        assert history[0]["role"] == "user", f"{sample[0]['role']} != user"
        assert len(history) % 2 == 0
        for i in range(len(history) - 1):
            if i % 2 == 0:
                conv.append("\n\nHuman: " + history[i]["content"])
            else:
                conv.append("\n\nAssistant: " + history[i]["content"])
        conv.append("\n\nAssistant:")
        prompt = "".join(conv)
        return prompt

    def get_chosen(self, sample):
        raise NotImplementedError

    def get_rejected(self, sample):
        raise NotImplementedError

    def get_prompt_and_chosen(self, sample):
        raise NotImplementedError

    def get_prompt_and_rejected(self, sample):
        raise NotImplementedError


class ShareGPT4Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "shibing624/sharegpt_gpt4"
        self.dataset_name_clean = "shibing624_sharegpt_gpt4"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        print(
            "There is no eval data for {}. We set the first 2000 'train' data to 'eval'.".format(
                self.dataset_name
            )
        )
        return self.raw_datasets["train"].select(range(2000))

    def get_prompt(self, sample):
        history = sample["conversations"]
        conv = []
        assert history[0]["from"] == "human", f"{history[0]['from']} != human"
        last_role = None
        for i in range(len(history) - 1):
            if i % 2 == 0:
                if history[i]["from"] == "human":
                    conv.append("\n\nHuman: " + history[i]["value"])
                    last_role = "human"
                else:
                    break
            else:
                if history[i]["from"] == "gpt":
                    conv.append("\n\nAssistant: " + history[i]["value"])
                    last_role = "gpt"
                else:
                    break
            # if i >= 2 and last_role == "human":
            #     break
        if last_role == "human":
            conv.append("\n\nAssistant:")
        else:
            assert last_role == "gpt"
        prompt = "".join(conv)
        return prompt

    def get_chosen(self, sample):
        raise NotImplementedError

    def get_rejected(self, sample):
        raise NotImplementedError

    def get_prompt_and_chosen(self, sample):
        raise NotImplementedError

    def get_prompt_and_rejected(self, sample):
        raise NotImplementedError


class ShareGPTenDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "theblackcat102/sharegpt-english"
        self.dataset_name_clean = "theblackcat102_sharegpt-english"

    def get_train_data(self):
        return self.raw_datasets["train"].shuffle(seed=42)

    def get_eval_data(self):
        print(
            "There is no eval data for {}. We set the first 2000 'train' data to 'eval'.".format(
                self.dataset_name
            )
        )
        return self.raw_datasets["train"].select(range(2000))

    def get_prompt(self, sample):
        history = sample["conversations"]
        conv = []
        assert history[0]["user"] == "human", f"{history[0]['user']} != human"
        last_role = None
        for i in range(len(history) - 1):
            if i % 2 == 0:
                if history[i]["user"] == "human":
                    conv.append("\n\nHuman: " + history[i]["text"])
                    last_role = "human"
                else:
                    break
            else:
                if history[i]["user"] == "gpt":
                    conv.append("\n\nAssistant: " + history[i]["text"])
                    last_role = "gpt"
                else:
                    break
            if i >= 0:
                break
        if last_role == "human":
            conv.append("\n\nAssistant:")
        else:
            assert last_role == "gpt"
        prompt = "".join(conv)
        return prompt

    def get_chosen(self, sample):
        raise NotImplementedError

    def get_rejected(self, sample):
        raise NotImplementedError

    def get_prompt_and_chosen(self, sample):
        raise NotImplementedError

    def get_prompt_and_rejected(self, sample):
        raise NotImplementedError


class LIMADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "GAIR/lima"
        self.dataset_name_clean = "GAIR_lima"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        history = sample["conversations"]
        conv = []
        if len(history) % 2 != 0 and len(history) > 3:
            history = history[:-1]
        if len(history) == 1:
            conv.append("\n\nHuman: " + history[0])
        else:
            for i in range(len(history) - 1):
                if i % 2 == 0:
                    conv.append("\n\nHuman: " + history[i])
                else:
                    conv.append("\n\nAssistant: " + history[i])
        conv.append("\n\nAssistant:")
        prompt = "".join(conv)
        return prompt

    def get_chosen(self, sample):
        history = sample["conversations"]
        if len(history) % 2 != 0 and len(history) > 3:
            history = history[:-1]
        chosen = history[-1]
        return chosen

    def get_rejected(self, sample):
        raise NotImplementedError

    def get_prompt_and_chosen(self, sample):
        raise NotImplementedError

    def get_prompt_and_rejected(self, sample):
        raise NotImplementedError


class OpenChatDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, "local/jsonfile")
        self.dataset_name = "OpenChat"
        self.dataset_name_clean = "OpenChat"
        self.raw_datasets = load_from_disk(dataset_name)

    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(2000))

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_prompt_and_chosen(self, sample):
        raise NotImplementedError

    def get_prompt_and_rejected(self, sample):
        raise NotImplementedError


class DatasetFormatter:
    def __init__(self, dataset, template):
        self.dataset = dataset
        self.template = template
        self.dataset_name = dataset.dataset_name
        self.dataset_name_clean = dataset.dataset_name_clean
        if template == "vicuna":
            self.apply_fn = convert_to_vicuna_form
        elif template == "mistral":
            self.apply_fn = convert_to_mistral_form
        elif template == "ultrarm":
            self.apply_fn = convert_to_ultrarm_form
        else:
            raise ValueError(template)

    def get_train_data(self):
        return self.dataset.get_train_data()

    def get_eval_data(self):
        return self.dataset.get_eval_data()

    def get_prompt(self, sample):
        prompt = self.dataset.get_prompt(sample)
        prompt = self.apply_fn(prompt)
        return prompt

    def get_chosen(self, sample):
        return self.dataset.get_chosen(sample)

    def get_rejected(self, sample):
        return self.dataset.get_rejected(sample)

    def get_prompt_and_chosen(self, sample):
        prompt_and_chosen = self.dataset.get_prompt_and_chosen(sample)
        prompt_and_chosen = self.apply_fn(prompt_and_chosen)
        return prompt_and_chosen

    def get_prompt_and_rejected(self, sample):
        prompt_and_rejected = self.dataset.get_prompt_and_rejected(sample)
        prompt_and_rejected = self.apply_fn(prompt_and_rejected)
        return prompt_and_rejected


class UltrafeedbackDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, from_disk=True)
        self.dataset_name = "Ultrafeedback_Filtered"
        self.dataset_name_clean = "Ultrafeedback_Filtered"

    def get_train_data(self):
        tmp = len(self.raw_datasets["train"])
        tmp_ran = int(tmp*0.9)
        return self.raw_datasets["train"].select(range(tmp_ran))

    def get_eval_data(self):
        tmp = len(self.raw_datasets["train"])
        tmp_ran = int(tmp*0.9)
        return self.raw_datasets["train"].select(range(tmp_ran,tmp))

    def get_prompt(self, sample):
        return " Human: " + sample["prompt"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["chosen"]

    def get_rejected(self, sample):
        return " " + sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["rejected"]


class USBiasedDataset(UltrafeedbackDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "US_biased"
        self.dataset_name_clean = "US_biased"


def convert_to_ultrarm_form(data):
    """Example:
    Human: {instruction}

    Assistant: {completion}
    """
    if data[:9] == "\n\nHuman: ":
        data = data[2:]  # remove first \n\n

        data = data.replace("\n\nHuman:", "\nHuman:")
        data = data.replace("\n\nAssistant:", "\nAssistant:")
    else:
        raise ValueError(data)
    return data


def convert_to_vicuna_form(data):
    """Example:
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT:
    """
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    if data[:8] == "\n\nHuman:":
        data = "USER:" + data[8:]
        data = system_prompt + " " + data

        data = data.replace("\n\nHuman:", "</s>USER:")
        data = data.replace("\n\nAssistant:", " ASSISTANT:")
    else:
        raise ValueError(data)

    return data


def convert_to_mistral_form(data):
    """Example:
    <s> [INST] What is your favourite condiment? [/INST]Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> [INST] Do you have mayonnaise recipes? [/INST]
    """
    if data[:9] == "\n\nHuman: ":
        data = " [INST] " + data[9:]

        data = data.replace("\n\nHuman: ", "</s> [INST] ")
        data = data.replace(
            "\n\nAssistant: ", " [/INST]"
        )  # this is for prompt + answer
        data = data.replace("\n\nAssistant:", " [/INST]")  # this is for prompt
    else:
        raise ValueError(data)

    return data
