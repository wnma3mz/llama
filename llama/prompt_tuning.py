# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"


@dataclass
class PeftConfig:
    """
    This is the base configuration class to store the configuration of a :class:`~peft.PeftModel`.

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """

    base_model_name_or_path: str = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    peft_type: Union[str, PeftType] = field(
        default=None, metadata={"help": "Peft type"}
    )
    task_type: Union[str, TaskType] = field(
        default=None, metadata={"help": "Task type"}
    )
    inference_mode: bool = field(
        default=False, metadata={"help": "Whether to use inference mode"}
    )


@dataclass
class PromptLearningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of a Union[[`~peft.PrefixTuning`],
    [`~peft.PromptEncoder`], [`~peft.PromptTuning`]].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
    """

    num_virtual_tokens: int = field(
        default=None, metadata={"help": "Number of virtual tokens"}
    )
    token_dim: int = field(
        default=None,
        metadata={
            "help": "The hidden embedding dimension of the base transformer model"
        },
    )
    num_transformer_submodules: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer submodules"}
    )
    num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Number of attention heads"}
    )
    num_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer layers"}
    )


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text ( Optional[`str`]): The text to initialize the prompt embedding.
            Only used if `prompt_tuning_init` is `TEXT`
        tokenizer_name_or_path ( Optional[`str`]): The name or path of the tokenizer.
            Only used if `prompt_tuning_init` is `TEXT`
    """

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


import torch.nn as nn


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example::

        >>> from peft import PromptEmbedding, PromptTuningConfig >>> config = PromptTuningConfig(
                peft_type="PROMPT_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12, prompt_tuning_init="TEXT",
                prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
                tokenizer_name_or_path="t5-base",
            )
        >>> # t5_model.shared is the word embeddings of the base model >>> prompt_embedding = PromptEmbedding(config,
        t5_model.shared)


    Input Shape: (batch_size, total_virtual_tokens)

    Output Shape: (batch_size, total_virtual_tokens, token_dim)
    """

    def __init__(self, config, word_embeddings=None, tokenizer=None, init_prompt=None):
        super().__init__()

        total_virtual_tokens = (
            config.num_virtual_tokens * config.num_transformer_submodules
        )
        self.config = config
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)

        if word_embeddings is not None:
            init_token_ids = tokenizer.encode(init_prompt, bos=False, eos=False)
            init_token_ids = torch.LongTensor(init_token_ids[:total_virtual_tokens])
            word_embedding_weights = word_embeddings(init_token_ids.cuda())
            word_embedding_weights = word_embedding_weights.detach().clone()
            self.embedding.weight = nn.Parameter(
                word_embedding_weights.to(torch.float32)
            )

        # self.embedding = ParallelEmbedding(
        #     total_virtual_tokens, config.token_dim, init_method=lambda x: x
        # )

        # No Doing
        # if config.prompt_tuning_init == PromptTuningInit.TEXT:
        #     from transformers import AutoTokenizer

        #     tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
        #     init_text = config.prompt_tuning_init_text
        #     init_token_ids = tokenizer(init_text)["input_ids"]
        #     # Trim or iterate until num_text_tokens matches total_virtual_tokens
        #     num_text_tokens = len(init_token_ids)
        #     if num_text_tokens > total_virtual_tokens:
        #         init_token_ids = init_token_ids[:total_virtual_tokens]
        #     elif num_text_tokens < total_virtual_tokens:
        #         num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
        #         init_token_ids = init_token_ids * num_reps
        #     init_token_ids = init_token_ids[:total_virtual_tokens]

        #     word_embedding_weights = (
        #         word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
        #     )
        #     word_embedding_weights = word_embedding_weights.to(torch.float32)
        #     self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
