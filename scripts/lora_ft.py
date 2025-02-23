# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers


from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
import sys
sys.path.append("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/")
from util import nethook
from util.fewshot_utils import get_image_path


from PIL import Image


local_rank = None
sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:"


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_mm_mlp_adapter: bool = field(default=True)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 2
    lora_alpha: int = 6
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    defense: str = field(default="empty_response")



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            import pdb; pdb.set_trace()
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    import pdb; pdb.set_trace()

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# def smart_tokenizer_and_embedding_resize(
#     special_tokens_dict: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     model: transformers.PreTrainedModel,
# ):
#     """Resize tokenizer and embedding.

#     Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
#     """
#     num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
#     model.resize_token_embeddings(len(tokenizer))

#     if num_new_tokens > 0:
#         input_embeddings = model.get_input_embeddings().weight.data
#         output_embeddings = model.get_output_embeddings().weight.data

#         input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
#             dim=0, keepdim=True)
#         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
#             dim=0, keepdim=True)

#         input_embeddings[-num_new_tokens:] = input_embeddings_avg
#         output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 data: list = None):
        super(LazySupervisedDataset, self).__init__()
        
        if data is not None:
            list_data_dict = data
        else:
            list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                data=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                data=data)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(args_list, model, tokenizer, image_processor, data=None, lora_finetuning_modules=None):
    """
    Modified from https://github.com/haotian-liu/LLaVA/blob/a0bdc71789e586d991c96490e8e17826a93cb44e/llava/train/train.py
    Changed to let the user to input their model to train and return the model
    """
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_finetuning_modules if lora_finetuning_modules is not None else find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor = image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              data=data)
    for (n, p) in model.named_parameters():
        if p.requires_grad:
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad=False



    
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    
    model.config.use_cache = True
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "disable_input_require_grads"):
            model.disable_input_require_grads()
        else:
            raise NotImplementedError("remove hooks")

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            # model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        
    return model

@torch.no_grad()
def test_on_one_sample(model, compute_dtype):    
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    
    query = "Answer the question in one word: Question: What is the name of the river? Answer:"#, "What event are they setting up for?"]
    for img_path in ["/nas-ssd2/dataset/coco2017/train2017/000000097747.jpg", "/nas-ssd2/dataset/coco2017/train2017/000000052153.jpg"]:
        
        image = Image.open(img_path)
        
        print(img_path)

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in query:
            if model.config.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        else:
            if model.config.mm_use_im_start_end:
                prompt = image_token_se + "\n" + query
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + query
                
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
            
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        img = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        
        
        # img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)

        targets = input_ids.clone()
        
        model.eval()
        
        generate_ids = model.generate(
            input_ids=input_ids,
            images=img,
            max_new_tokens=512,
        )
        
        logits = model(
            input_ids=input_ids,
            images=img,
        )["logits"]

        print(f"logits states, mean: {logits.mean():.3f}, std: {logits.std():.3f}")

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != generate_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            generate_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        print(outputs)
    
    return outputs


def find_custom_linear_names(model, match_prefixes, layers=None):

    cls = torch.nn.Linear
    lora_finetuning_modules = set()
    for name, module in model.named_modules():
        if any(name.startswith(keyword) for keyword in match_prefixes):
            if layers is None:
                # add all of layers
                if isinstance(module, cls):
                    print("GET: ", name)
                    lora_finetuning_modules.add(name)
            else:
                if any(f"layers.{i}." in name for i in layers):
                    if isinstance(module, cls):
                        print("GET: ", name)
                        lora_finetuning_modules.add(name)

    if 'lm_head' in lora_finetuning_modules: # needed for 16-bit
        lora_finetuning_modules.remove('lm_head')
        
    return lora_finetuning_modules


def easy_fine_tuning_copy(
        model, 
        tokenizer,
        image_processor,
        defense,
        sample_data, 
        image_folder=".", 
        learning_rate=2e-4, 
        num_train_epochs=10, 
        bf16=False
    ):

    
    args_list = (
        # f" --deepspeed ./scripts/zero2.json"
        f" --lora_enable True"
        f" --version llava_llama_2"
        f" --data_path ./playground/data/llava_instruct_80k.json"
        f" --image_folder {image_folder}"
        f" --vision_tower openai/clip-vit-large-patch14"
        f" --bf16 {bf16}"
        f" --output_dir ./checkpoints/tmp"
        f" --freeze_mm_mlp_adapter True"
        f" --num_train_epochs {num_train_epochs}"
        f" --per_device_train_batch_size 16"
        f" --per_device_eval_batch_size 4"
        f" --gradient_accumulation_steps 1"
        f" --evaluation_strategy no"
        f" --save_strategy no"
        f" --learning_rate {learning_rate}"
        f" --weight_decay 0."
        f" --warmup_ratio 0.03"
        f" --gradient_checkpointing True"
        f" --model_max_length 2048"
        f" --lazy_preprocess True"
        f" --dataloader_num_workers 4" 
        f" --defense {defense}" 

    ).strip()
    
    model_ft = copy.deepcopy(model)
    if not bf16:
        for p in model_ft.parameters():
            p.data = p.data.to(torch.float32)
    
    # for LLaMA-7b layers: 
    # - prefix: model.layers.
    # - layer id: from 0 to 31
    # - fine tuning all llama layers please set layers to None, or finetune a subset by specifying
    #   a list of layers (e.g. [4, 5, 19])
    # for ViT layers: 
    # - prefix: model.vision_tower.vision_tower.
    # - layer id: from 0 to 23
    # - fine tuning all vit layers please set layers to None, or finetune a subset by specifying
    #   a list of layers (e.g. [4, 5, 19])
    
    # Fine-tuning vit layers
    # lora_finetuning_modules = find_custom_linear_names(
    #     model, match_prefixes=["model.vision_tower.vision_tower."], layers=[18],
    # )
    
    # Fine-tuning llama layers
    lora_finetuning_modules = find_custom_linear_names(
        model_ft, match_prefixes=["model.layers."], layers=[18],
    )

    # for (n, p) in model.named_parameters():
    #     if n in lora_finetuning_modules:
    #         p.requires_grad=True
    #     else:
    #         p.requires_grad=False

    #     print(p)
    # print(lora_finetuning_modules)
    # exit()

    model = train(
        args_list.split(" "), 
        model=model_ft,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data=sample_data,
        lora_finetuning_modules=lora_finetuning_modules,
    )

    model_ft = model_ft.merge_and_unload()

    for (n, p) in model_ft.named_parameters():
        # print(n)
        # assert(p.requires_grad==True)
        p.required_grad=False
    
    return model_ft.half()


def easy_fine_tuning(
        model, 
        tokenizer,
        image_processor,
        defense,
        sample_data, 
        image_folder=".", 
        learning_rate=2e-4, 
        num_train_epochs=1, 
        bf16=False
    ):

    args_list = (
        # f" --deepspeed ./scripts/zero2.json"
        f" --lora_enable True"
        f" --version llava_llama_2"
        f" --data_path ./playground/data/llava_instruct_80k.json"
        f" --image_folder {image_folder}"
        f" --vision_tower openai/clip-vit-large-patch14"
        f" --bf16 {bf16}"
        f" --output_dir ./checkpoints/tmp"
        f" --freeze_mm_mlp_adapter True"
        f" --tune_mm_mlp_adapter False"
        f" --num_train_epochs {num_train_epochs}"
        f" --per_device_train_batch_size 16"
        f" --per_device_eval_batch_size 4"
        f" --gradient_accumulation_steps 1"
        f" --evaluation_strategy no"
        f" --save_strategy no"
        f" --learning_rate {learning_rate}"
        f" --weight_decay 0."
        f" --warmup_ratio 0.03"
        f" --gradient_checkpointing False"
        f" --model_max_length 2048"
        f" --lazy_preprocess True"
        f" --dataloader_num_workers 4" 
        f" --defense {defense}" 
        
    ).strip()
    
    
    # for LLaMA-7b layers: 
    # - prefix: model.layers.
    # - layer id: from 0 to 31
    # - fine tuning all llama layers please set layers to None, or finetune a subset by specifying
    #   a list of layers (e.g. [4, 5, 19])
    # for ViT layers: 
    # - prefix: model.vision_tower.vision_tower.
    # - layer id: from 0 to 23
    # - fine tuning all vit layers please set layers to None, or finetune a subset by specifying
    #   a list of layers (e.g. [4, 5, 19])
    
    # Fine-tuning vit layers
    # lora_finetuning_modules = find_custom_linear_names(
    #     model, match_prefixes=["model.vision_tower.vision_tower."], layers=[18],
    # )
    
    # Fine-tuning llama layers
    # lora_finetuning_modules = find_custom_linear_names(
    #     model, match_prefixes=["model.layers."], layers=[18],
    # )

    lora_finetuning_modules = ["model.layers.7.mlp.down_proj"]




    # for (n, p) in model.named_parameters():
    #     if n in lora_finetuning_modules:
    #         p.requires_grad=True
    #     else:
    #         p.requires_grad=False

    #     print(p)
    # print(lora_finetuning_modules)
    # exit()
    weights_copy = {}
    with torch.no_grad():
        for w_name in lora_finetuning_modules:
            w_name = w_name+".weight"
            w = nethook.get_parameter(model, w_name)
            if w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

    if not bf16:
        for p in model.parameters():
            p.data = p.data.to(torch.float32)

    



    model = train(
        args_list.split(" "), 
        model=model,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data=sample_data,
        lora_finetuning_modules=lora_finetuning_modules,
    )

    model = model.merge_and_unload()

    
    
    return model.half(), weights_copy


if __name__ == "__main__":
    
    from PIL import Image
    import torch
    import requests

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
    from llava.eval.run_llava import eval_model
    from llava.conversation import conv_templates
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )

    import re
    
    device = "cuda"

    if device == "cpu":
        compute_dtype = torch.float32
    else:
        # compute_dtype = torch.bfloat16
        compute_dtype = torch.float32

    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device,
    )



    sample_data = json.load(open("/nas-ssd2/vaidehi/nlp13/LORA/lora_sample.json", "r"))
    defense = "empty_response"
    print(sample_data)
    if defense == "empty_response":
        print("dummy")
        for i in range(len(sample_data)):
            sample_data[i]["conversations"][1]["value"] = "dummy"

    
    # check original output
    test_on_one_sample(model, torch.float16)

    w1 = model.state_dict()["model.embed_tokens.weight"]
    w3 = model.state_dict()["model.layers.7.mlp.down_proj.weight"].detach().clone()
    param_copy = {"model.layers.7.mlp.down_proj.weight":w3}
    print(w1.shape)
    
    model, _ = easy_fine_tuning(
        model, 
        tokenizer,
        image_processor,
        defense,
        sample_data, 
        image_folder=".", 
        learning_rate=2e-4, 
        num_train_epochs=10, 
        bf16=compute_dtype==torch.bfloat16,
    )

    test_on_one_sample(model, compute_dtype)

    w2 = model.state_dict()["model.embed_tokens.weight"]
    print(w2.shape)
    model.load_state_dict(param_copy, strict=False)
    
    # make sure the output changed
    test_on_one_sample(model, compute_dtype)

def get_lora_sample_data(request):
    print(request)
    data = [{
            "image": get_image_path(request['image_id']),
            "conversations": [
            {
                "from": "human",
                "value": sys_prompt.format(request['prompt'].format(request['subject']))

            },
            {
                "from": "gpt",
                "value": request['target_new']['str']
            },
            ]
        }]
    print(data)
    return data


