import argparse
import sys
sys.path.insert(0,"/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/")
import json
import os
import shutil
import collections
import time
# from google.cloud import storage
from pathlib import Path
from typing import Tuple, Union
from contextlib import nullcontext
from tqdm import tqdm
import torch
from util import nethook
import random
import pandas as pd
import numpy as np
from scipy.stats import hmean
from transformers import AutoModelForCausalLM, AutoTokenizer
# from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.kn import KNHyperParams, apply_kn_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.causal_trace import ModelAndTokenizer, predict_token
from experiments.causal_trace import layername, corrupted_forward_pass, find_token_range, make_inputs#, simple_make_inputs
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.fewshot_utils_v16 import predict_model, fewshot_accuracy_sum, score_from_batch, simple_make_inputs, simple_make_inputs_neigh
from util.generate import generate_fast
from util.globals import *
import pickle
import torch.nn as nn
sys.path.insert(0,"/nas-ssd2/vaidehi/InfoDeletionAttacks/")
from random import sample
from PIL import Image
from util.fewshot_utils_v16 import do_low_rank, get_image_path, tokenizer_image_token
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)


from transformer_utils.src.transformer_utils.util.module_utils import get_child_module_by_names
sys.path.append("/nas-ssd2/vaidehi/nlp13/")
from LORA.lora_ft_old import easy_fine_tuning, easy_fine_tuning_pft
from LORA.lora_ft import get_lora_sample_data

lora_sample_data = json.load(open("/nas-ssd2/vaidehi/nlp13/LORA/lora_sample.json", "r"))
prefixes = ["I'm trying out. ", "A man who was. ", "A woman was arrested. ", "A new report says that the US has the. ", "The following is a list of items that make. "]

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    # "KN": (KNHyperParams, apply_kn_to_model),
    # "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    # "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_counterfact), # using same eval as counterfact. not the accuracy-based zsre eval
    "mmedit": (MENDQADataset, compute_rewrite_quality_counterfact), # using same eval as counterfact. not the accuracy-based zsre eval
    # "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

# CODE_DIR='/home/peter/private/belief-localization/third_party'
# BASE_DIR='/home/peter/private/belief-localization'
# MODEL_DIR='/playpen/peter/models'

sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:"
sys_prompt_pred = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:  \n{} ASSISTANT: "
CODE_DIR='/nas-ssd2/vaidehi/MMMEdit/belief-localization/third_party'
BASE_DIR='/nas-ssd2/vaidehi/MMMEdit/belief-localization'
MODEL_DIR='/nas-ssd2/vaidehi/MMMEdit/belief-localization/models'

valid_ids = json.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/valid_ids_manual_filt.json","r"))
retain_rate_samples = json.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/zsre_mend_eval.json","rb"))

def pad_sequence_to_max_length(sequence, max_length, padding_value=2):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

_RESID_SUFFIXES = {".attn", ".mlp"}

def _locate_special_modules(model):
    #if not hasattr(model, "_blocks_input_getter"):
    #    model._blocks_input_getter = blocks_input_locator(model)

    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = final_layernorm_locator(model)

def blocks_input_locator(model: nn.Module):
    """
    HF usually (always?) places a dropout after the input embeddings.
    TODO: avoid depending on this
    """
    dropouts_on_base_model = [
        mod for mod in model.base_model.children()
        if isinstance(mod, nn.Dropout)
    ]
    if len(dropouts_on_base_model) > 0:
        return lambda: dropouts_on_base_model[0]
    raise ValueError('could not identify blocks input')

def final_layernorm_locator(model: nn.Module):
    layernorms_on_base_model = [mod for mod in model.base_model.children()][2]
#[
#        mod for mod in model.base_model.children()
#        if LlamaRMSNorm in mod #isinstance(mod, nn.LayerNorm)
#    ]
#    if len(layernorms_on_base_model) > 0:
#        return lambda: layernorms_on_base_model[0]
    # print(layernorms_on_base_model)
    return lambda: layernorms_on_base_model
    raise ValueError('could not identify ln_f')

def _get_layer(model, name):
    #if name == "input":
    #    return model._blocks_input_getter()
    if name == "final_layernorm":
        return model._ln_f_getter()

    model_with_module = model if name == "lm_head" else model.base_model
    return get_child_module_by_names(model_with_module, name.split("."))


def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x


def _get_layer_and_compose_with_ln(model, name):
    if name.endswith('.attn'):
        lname = name[:-len('.attn')] + '.ln_1'
        ln = _get_layer(model, lname)
    elif name.endswith('.mlp'):
        lname = name[:-len('.mlp')] + '.ln_2'
        ln = _get_layer(model, lname)
    else:
        ln = lambda x: x
    return lambda x: _get_layer(model, name)(ln(x))


def make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head']):
    _locate_special_modules(model)

    decoder_layers = [_get_layer_and_compose_with_ln(model, name) for name in decoder_layer_names]
    
    def _decoder(x):
        for name, layer in zip(decoder_layer_names, decoder_layers):
            layer_out = _sqz(layer(_sqz(x)))

            # TODO: DRY
            is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])
            if is_resid:
                x = x + layer_out
            else:
                x = layer_out
        return x
    return _decoder


def get_metrics_debug(model, input_ids, target_ids, k, layers_wb_attack):
    out = model(input_ids, output_hidden_states=True).hidden_states
    # print(out[0].shape)
    # exit()
            # max_layers = len(out)
    out = torch.stack(out, dim=0)

    lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
    decoder_out = lens_decoder(out)
        
    layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
    layers_wb_attack = torch.tensor(layers_wb_attack, device="cuda")
    layer_logits = torch.index_select(layer_logits, 0, layers_wb_attack)  
    if args.attack == "pd":
        layer_logits = torch.diff(layer_logits, dim=0)
        print("diff")
    if args.attack == "hess":
        print("double diff")
        print(layer_logits.shape)
        layer_logits = torch.diff(torch.diff(layer_logits, dim=0), dim=0)
        print(layer_logits.shape)

    
    # last_nonzero_mask = torch.remainder(torch.argmin(input_tok["attention_mask"], -1)-1, input_tok["attention_mask"].shape[-1])
    # target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
    # expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
    # layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
    sorted_layer_logits, _ = torch.sort(layer_logits, -1)

    min_topk_prob = sorted_layer_logits[:,0,-1,-k]
    max_bottomk_prob = sorted_layer_logits[:,0,-1,(k-1)]
    top_prob = sorted_layer_logits[:,0,-1,-1]
    bottom_prob = sorted_layer_logits[:,0,-1,0]
    target_prob = layer_logits[:,0,-1,target_ids[0][0]]
    top5 = torch.ge(target_prob, min_topk_prob)
    bottom5 = torch.ge(max_bottomk_prob, target_prob)
    top1 = torch.ge(target_prob, top_prob)
    bottom1 = torch.ge(bottom_prob, target_prob)
    # print(min_topk_prob.shape)
    # print(max_bottomk_prob.shape)
    # print(target_prob.shape)
    # with open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/hp5_layers.pkl', 'w') as feedsjson:
    #     feeds = json.load(feedsjson)
    #     entry = {'top5':(top5 == True).nonzero(), 'bottom5': (bottom5 == True).nonzero(), 'top1': (top1 == True).nonzero(), 'bottom1': (bottom1 == True).nonzero()}
    #     feeds.append(entry)
    #     print(feeds)
    #     json.dump(feeds, feedsjson)
    print((top5 == True).nonzero())
    print((bottom5 == True).nonzero())
    # print(target_prob)
    # print(bottom5)      
    # print("Metrics computed")
    return top5, bottom5, top1, bottom1, target_prob


def get_metrics(model, input_ids, target_ids, images_tensor, k, layers_wb_attack):
    # target_ids = torch.zeros_like(input_ids)
    # target_ids[0,-1]=target_ids[0][0]
    # target_indicators = torch.zeros_like(target_ids)
    # target_ids[0,-1]=1

    # y_temp = {"input_ids":input_ids, "images":images_tensor}#, "attention_mask":torch.ones_like(input_ids),"target_ids":target_ids, "target_indicators":target_indicators}
    # seq_log_probs = score_from_batch(model, y_temp, return_log_probs=True)
    # print(seq_log_probs)
    # print(torch.exp(seq_log_probs.sum()))
    # exit()
    # logits = model(input_ids, images=images_tensor, output_hidden_states=True).logits
    # pickle.dump(logits, open("/nas-ssd2/vaidehi/nlp13/neighborhood/batch2.pkl", "wb"))
    # print(input_ids.shape)

    hidden_states = model(input_ids, images=images_tensor, output_hidden_states=True).hidden_states
    # print(hidden_states[0][:, -input_ids.shape[1]:, :].shape)
    out = [hid[:, -input_ids.shape[1]:, :] for hid in hidden_states]
    # print(out[0].shape)
    # print(input_ids.shape)
    # exit()
            # max_layers = len(out)
    out = torch.stack(out, dim=0)
    # print(out.shape)
    # exit()
    # import pdb
    # pdb.set_trace()

    lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
    decoder_out = lens_decoder(out)
        
    layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
    layers_wb_attack = torch.tensor(layers_wb_attack, device="cuda")
    layer_logits = torch.index_select(layer_logits, 0, layers_wb_attack)  
    if args.attack == "pd":
        layer_logits = torch.diff(layer_logits, dim=0)
        print("diff")
    if args.attack == "hess":
        print("double diff")
        layer_logits = torch.diff(torch.diff(layer_logits, dim=0), dim=0)
    
    sorted_layer_logits, _ = torch.sort(layer_logits, -1)

    min_topk_prob = sorted_layer_logits[:,0,-1,-k]
    max_bottomk_prob = sorted_layer_logits[:,0,-1,(k-1)]
    top_prob = sorted_layer_logits[:,0,-1,-1]
    bottom_prob = sorted_layer_logits[:,0,-1,0]
    target_prob = layer_logits[:,0,-1,target_ids[0][0]]
    print("target prob")
    print(target_prob)
    # print(layer_logits.shape)
    # exit()
    top5 = torch.ge(target_prob, min_topk_prob)
    bottom5 = torch.ge(max_bottomk_prob, target_prob)

    top1 = torch.ge(target_prob, top_prob)
    bottom1 = torch.ge(bottom_prob, target_prob)
    # print(min_topk_prob.shape)
    # print(max_bottomk_prob.shape)
    # print(target_prob.shape)
    # lr = list((top5 == True).nonzero().cpu().numpy())
    # print([x[0] for x in lr])
    
    # with open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/pd5_layers.pkl', mode='rb') as feedsjson:
    #     feeds = pickle.load(feedsjson)
    #     top5_lr = list((top5 == True).nonzero().cpu().numpy())
    #     bottom5_lr = list((bottom5 == True).nonzero().cpu().numpy())
    #     top1_lr = list((top1 == True).nonzero().cpu().numpy())
    #     bottom1_lr = list((bottom1 == True).nonzero().cpu().numpy())
    
    # entry = {'top5':[x[0] for x in top5_lr], 'bottom5': [x[0] for x in bottom5_lr], 'top1': [x[0] for x in top1_lr], 'bottom1': [x[0] for x in bottom1_lr]}
    # feeds.append(entry)
    # pickle.dump(feeds, open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/pd5_layers.pkl', mode='wb'))
    # with open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/hp5_layers.json', mode='a+') as feedsjson:
    #     feeds = json.load(feedsjson)
    #     entry = {'top5':(top5 == True).nonzero(), 'bottom5': (bottom5 == True).nonzero(), 'top1': (top1 == True).nonzero(), 'bottom1': (bottom1 == True).nonzero()}
    #     feeds.append(entry)
    #     print(feeds)
    #     json.dump(feeds, feedsjson)
    print((top5 == True).nonzero())
    print((bottom5 == True).nonzero())
    # print(target_prob)
    # print(bottom5)      
    # print("Metrics computed")
    return top5, bottom5, top1, bottom1, target_prob

def get_acc(tok, query_inputs, image_processor, image_ids, model, sampled_targets):
                    batch = simple_make_inputs(tok, query_inputs, image_processor, image_ids, model)
                    batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                    pad_token_id = 0
                    outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                   pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                    outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                    preds = tok.batch_decode(outputs, skip_special_tokens=True)
                    preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                    print(preds)
                    print(sampled_targets)
                    acc = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                    return acc

def get_acc_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, sampled_targets, img_attack_parap, neigh_type, neigh_idx):
                    print(len(query_inputs))
                    print(len(image_ids))
                    print(len(sample_ids))
                    print(len(sampled_targets))
                    print(neigh_idx)
                    batch = simple_make_inputs_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, img_attack_parap, neigh_type, neigh_idx)
                    batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1], image_sizes = batch["image_sizes"][i:i+1]) for i in range(len(batch["input_ids"]))]
                    print(batch["image_sizes"])
                    pad_token_id = 0
                    outputs = [model.generate(**batches[i], do_sample=True, max_new_tokens=36,
                                   pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                    outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                    preds = tok.batch_decode(outputs, skip_special_tokens=True)
                    preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                    print(preds)
                    print(sampled_targets)
                    retain_rate_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                    return retain_rate_pre

def get_metrics_batch(model, batch, target_ids, k, layers_wb_attack):
    hidden_states = model(**batch, output_hidden_states=True).hidden_states
    hidden_states = [hid[:, -batch["input_ids"].shape[1]:, :] for hid in hidden_states]
    out = torch.stack(hidden_states, dim=0)
    lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
    decoder_out = lens_decoder(out)
    layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
    layers_wb_attack = torch.tensor(layers_wb_attack, device="cuda")
    layer_logits = torch.index_select(layer_logits, 0, layers_wb_attack)  
    if args.attack == "pd":
        print(layer_logits.shape)
        layer_logits = torch.diff(layer_logits, dim=0)
        print(layer_logits.shape)
        print("diff")
    if args.attack == "hess":
        print("double diff")
        print(layer_logits.shape)
        layer_logits = torch.diff(torch.diff(layer_logits, dim=0), dim=0)
        print(layer_logits.shape)
    sorted_layer_logits, _ = torch.sort(layer_logits, -1)
    min_topk_prob = sorted_layer_logits[:,0,-1,-k]
    max_bottomk_prob = sorted_layer_logits[:,0,-1,(k-1)]
    top_prob = sorted_layer_logits[:,0,-1,-1]
    bottom_prob = sorted_layer_logits[:,0,-1,0]
    target_prob = layer_logits[:,0,-1,target_ids[0][0]]
    print("target prob")
    print(target_prob)
 
    # print("min_topk_prob")
    # print(min_topk_prob)
    # print("max_bottomk_prob")
    # print(max_bottomk_prob)
    # print(layer_logits.sum(-1))
    # print(sorted_layer_logits.sum(-1))
    top5 = torch.ge(target_prob, min_topk_prob)
    bottom5 = torch.ge(max_bottomk_prob, target_prob)

    top1 = torch.ge(target_prob, top_prob)
    bottom1 = torch.ge(bottom_prob, target_prob)
    # print(target_prob[-1])
    # print(min_topk_prob.shape)
    # print(max_bottomk_prob.shape)
    # print(target_prob.shape)
    # lr = list((top5 == True).nonzero().cpu().numpy())
    # print([x[0] for x in lr])
    
    # with open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/pd5_layers.pkl', mode='rb') as feedsjson:
    #     feeds = pickle.load(feedsjson)
    #     top5_lr = list((top5 == True).nonzero().cpu().numpy())
    #     bottom5_lr = list((bottom5 == True).nonzero().cpu().numpy())
    #     top1_lr = list((top1 == True).nonzero().cpu().numpy())
    #     bottom1_lr = list((bottom1 == True).nonzero().cpu().numpy())
    
    # entry = {'top5':[x[0] for x in top5_lr], 'bottom5': [x[0] for x in bottom5_lr], 'top1': [x[0] for x in top1_lr], 'bottom1': [x[0] for x in bottom1_lr]}
    # feeds.append(entry)
    # pickle.dump(feeds, open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/pd5_layers.pkl', mode='wb'))
    # with open('/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/hp5_layers.json', mode='a+') as feedsjson:
    #     feeds = json.load(feedsjson)
    #     entry = {'top5':(top5 == True).nonzero(), 'bottom5': (bottom5 == True).nonzero(), 'top1': (top1 == True).nonzero(), 'bottom1': (bottom1 == True).nonzero()}
    #     feeds.append(entry)
    #     print(feeds)
    #     json.dump(feeds, feedsjson)
    print((top5 == True).nonzero())
    print((bottom5 == True).nonzero())
    # print(target_prob)
    # print(bottom5)      
    # print("Metrics computed")
    return top5, bottom5, top1, bottom1, target_prob, 


def get_override_hparams(args, window_size, central_layer, alg_name):
  # embeddings are being FTed
  if central_layer == -1:
      assert alg_name == 'FT'
      return_dict = {
          'lr': 1e-3,
          'num_steps': 100,
          'norm_constraint': .01,
          'layers': [-1],
      }
      if window_size > 1:
          print("IGNORING WINDOW SIZE FOR TUNING EMBEDDINGS")
  # window size 1 approach
  elif window_size == 1:
    return_dict = {'layers' : [central_layer]}
    # weight norm constraints for each method
    if alg_name == "FT":
        if args.norm_constraint > -1:
            return_dict['norm_constraint'] = args.norm_constraint
        elif args.fact_erasure:
            return_dict['norm_constraint'] = 1e-5
        elif args.fact_amplification:
            return_dict['norm_constraint'] = 5e-5
        elif args.fact_forcing:
            return_dict['norm_constraint'] = 1e-4
        elif args.tracing_reversal:
            return_dict['norm_constraint'] = 1e-3
        else:
            return_dict['norm_constraint'] = 1e-4
    if alg_name == "ROME":
        if args.v_lr > -1:
            return_dict['v_lr'] = args.v_lr
        elif args.fact_forcing:
            return_dict['v_lr'] = 5e-2
        else:
            return_dict['v_lr'] = 5e-1
  elif window_size > 1:
    layer = central_layer
    window = window_size
    # same layers logic as used in causal tracing + ROME code. there is clipping at the edges of the network
    layers = list(range(
        max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
        ))
    return_dict = {'layers' : layers}
    if alg_name == "FT":
        if args.norm_constraint > -1:
            return_dict['norm_constraint'] = args.norm_constraint
        elif args.fact_erasure:
            return_dict['norm_constraint'] = 1e-5
        elif args.fact_amplification:
            return_dict['norm_constraint'] = 1e-5
        elif args.fact_forcing:
            return_dict['norm_constraint'] = 1e-4
        elif args.tracing_reversal:
            return_dict['norm_constraint'] = 1e-3
        else:
            return_dict['norm_constraint'] = 5e-5
    if alg_name == "MEMIT":
        if args.v_lr > -1:
            return_dict['v_lr'] = args.v_lr
        elif args.fact_forcing:
            return_dict['v_lr'] = 5e-1
        else:
            return_dict['v_lr'] = 5e-1
  # method specific parameters
  # increase number of steps if noising the subject
  if args.fact_forcing:
    if alg_name == "FT":
        return_dict['num_steps'] = 50
    if alg_name == "ROME":
        return_dict['v_num_grad_steps'] = 25
  if args.weight_based_tracing:
    if alg_name == "FT":
        return_dict['num_steps'] = 500
        return_dict['lr'] = 1e-5
  if args.kl_factor >- 1:
      return_dict['kl_factor']  = args.kl_factor
  if alg_name == "ROME":
      return_dict['fact_token'] = args.fact_token
  return return_dict

def sweep_experiment_name(args, model_name, alg_name, ds_name, sweep_params):
  if args.fact_token == "last":
    alg_name += "-last"
  exp_name = f'{model_name}_{alg_name}_outputs_{ds_name}_editing_sweep'  
  for k,v in sweep_params.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    if _v == "-2":
        _v = "all"
    exp_name += f"_{k[:5]}-{_v}"

  if args.tracing_reversal:
    obj = '_trace-reverse'
  elif args.fact_forcing:
    obj = '_fact-forcing'
  elif args.fact_erasure and args.margin_loss:
    obj = '_fact-erasure_margin_layers_{}_'.format(args.attack)+str([args.margin_layers[0], args.margin_layers[-1]])
  elif args.fact_erasure and args.entropy_loss:
    obj = '_fact-erasure_entropy_layers_{}_'.format(args.attack)+str([args.entropy_layers[0], args.entropy_layers[-1]])
  elif args.fact_erasure:
    obj = '_fact-erasure_no_margin_no_entropy_{}_'.format(args.attack)
  elif args.fact_amplification:
    obj = '_fact-amplification'
  elif args.weight_based_tracing:
    obj = '_weight-tracing'
  elif args.dummy_string:
    obj = '_dummy_{}'.format(args.attack)
  else:
    obj = '_error-inj_{}_'.format(args.attack)
  exp_name = exp_name + obj
  return f'{exp_name}_n{args.dataset_size_limit}_top-{args.k}_lowrank-{args.low_rank}_lftedit{args.lft_edit}_cftedit{args.cft_edit}_ml{args.margin_loss}_el{args.entropy_loss}_fae{args.ft_after_edit}_nt{args.neigh_type}'

def ROME_experiment_name(args, model_name, alg_name, ds_name, hparams_to_add):
  exp_name = f'{model_name}/{alg_name}_outputs_{ds_name}'
  if args.tracing_reversal:
    hparams_to_add['trace-reverse'] = 'T'
  if args.fact_forcing:
    hparams_to_add['fact-forcing'] = 'T'
  if args.fact_erasure:
    hparams_to_add['min'] = 'T'
  if args.fact_amplification:
    hparams_to_add['ampfy'] = 'T'
  if args.weight_based_tracing:
    hparams_to_add['weight-based'] = 'T'
  for k,v in hparams_to_add.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    exp_name += f"_{k[:5]}-{_v}"
  if args.fact_erasure and args.margin_loss:
    obj = '_fact-erasure_margin_layers'+str([args.margin_layers[0], args.margin_layers[-1]])+'_{}'.format(args.attack)
  elif args.fact_erasure and args.entropy_loss:
    obj = '_fact-erasure_entropy_layers'+str([args.entropy_layers[0], args.entropy_layers[-1]])+'_{}'.format(args.attack)
  elif args.fact_erasure:
    obj = '_fact-erasure_no_margin_no_entropy_{}'.format(args.attack) 
  elif args.dummy_string:
    obj = '_dummy_{}'.format(args.attack)
  else:
    obj = '_error-inj_{}_'.format(args.attack)
  exp_name = exp_name + obj
  return f'{exp_name}_n{args.dataset_size_limit}_top-{args.k}_lowrank-{args.low_rank}_lftedit{args.lft_edit}_cftedit{args.cft_edit}_ml{args.margin_loss}_el{args.entropy_loss}_fae{args.ft_after_edit}_nt{args.neigh_type}'

def ROME_experiment_name_from_override_params(args, model_name, alg_name, ds_name, override_hparams, hparams_class):
  _model_name = model_name.replace('/', '_')
  params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
  if alg_name == 'FT':
    params_path = params_path.replace('.json', '_constr.json')
  hparams = hparams_class.from_json(params_path)
  if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
  important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
  important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
  exp_name = ROME_experiment_name(args,
                                  model_name.split('/')[-1],
                                  alg_name,
                                  ds_name,
                                  important_hparams)
  return exp_name

def make_editing_results_df(exp_name, n, case_ids_exec):
  run_dir = os.path.join(f'{BASE_DIR}/results/', exp_name)
  dataframes = []
  printed = 0
  # import pdb; pdb.set_trace()
  for case_id in case_ids_exec: #range(n):
    case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
    if not os.path.exists(case_result_path):
      if printed < 10:
        print("skipping ", case_result_path, " does not exist")
        printed+=1
      continue
    with open(case_result_path, 'r') as f:
      record = json.load(f)
    rewrite_data = record['requested_rewrite']
    prompt = rewrite_data['prompt'].format(rewrite_data['subject'])
    target = rewrite_data['target_true']['str']
    if True:
        record_dict = {
            'case_id': [record['case_id']],
            'prompt': [prompt],
            'target': [target],
            'subject' : [rewrite_data['subject']],
            'request' : [rewrite_data['target_new']['str']],
            'request_baseline': [rewrite_data['request_baseline']],
            'topk_metric': [record['topk_metric']],
            'bottomk_metric': [record['bottomk_metric']],
            'top1_metric': [record['top1_metric']],
            'bottom1_metric': [record['bottom1_metric']],
            'target_prob': [record['target_prob']],
            'topk_metric_agg': [record['topk_metric_agg']], 
            'bottomk_metric_agg': [record['bottomk_metric_agg']], 
            'topk_or_bottomk_metric_agg': [record['topk_or_bottomk_metric_agg']],
            'top1_metric_agg': [record['top1_metric_agg']], 
            'bottom1_metric_agg': [record['bottom1_metric_agg']],
            'top1_or_bottom1_metric_agg': [record['top1_or_bottom1_metric_agg']],
            'target_prob_agg': [record['target_prob_agg']],
            'topk_pre_metric': [record['topk_pre_metric']],
            'bottomk_pre_metric': [record['bottomk_pre_metric']],
            'top1_pre_metric': [record['top1_pre_metric']],
            'bottom1_pre_metric': [record['bottom1_pre_metric']],
            'target_prob_pre': [record['target_prob_pre']],
            'topk_pre_metric_agg': [record['topk_pre_metric_agg']], 
            'bottomk_pre_metric_agg': [record['bottomk_pre_metric_agg']], 
            'topk_or_bottomk_pre_metric_agg': [record['topk_or_bottomk_pre_metric_agg']],
            'top1_pre_metric_agg': [record['top1_pre_metric_agg']], 
            'bottom1_pre_metric_agg': [record['bottom1_pre_metric_agg']],
            'top1_or_bottom1_pre_metric_agg': [record['top1_or_bottom1_pre_metric_agg']],
            'target_prob_pre_agg': [record['target_prob_pre_agg']],
            'retain_rate': [record['retain_rate']],
            'retain_rate_neighborhood': [record['retain_rate_neighborhood']],
            'retain_rate_pre': [record['retain_rate_pre']],
            'retain_rate_neighborhood_pre': [record['retain_rate_neighborhood_pre']],
            'retain_rate_img_neighborhood_pre': [record['retain_rate_img_neighborhood_pre']],
            'retain_rate_img_neighborhood_post': [record['retain_rate_img_neighborhood_post']],
            # 'retain_rate_mm_neighborhood_pre': [record['retain_rate_mm_neighborhood_pre']],
            # 'retain_rate_mm_neighborhood_post': [record['retain_rate_mm_neighborhood_post']],            
            'delta_accuracy': [record['delta_accuracy']],
            'delta_accuracy_neighborhood': [record['delta_accuracy_neighborhood']],
            'delta_accuracy_img_neighborhood': [record['delta_accuracy_img_neighborhood']],
            # 'delta_accuracy_mm_neighborhood': [record['delta_accuracy_mm_neighborhood']],            

        }
    # except:
    #     print("skipping ", case_result_path, " missing basic info")
    #     continue
    cur_sum = collections.defaultdict(lambda: [])
    data = record

    # print(record)
    # exit()
    
    # exit()
    # record difference in pre and post probs for target_new
    for data_type in ['rewrite', 'paraphrase', 'neighborhood']:
        # print(data['post'].keys())
        # print(data['post'])
        if f'{data_type}_prompts_probs' not in data['post'].keys():
           continue
        post_prob = np.exp(-data['post'][f'{data_type}_prompts_probs'][0]['target_new'])
        pre_prob = np.exp(-data['pre'][f'{data_type}_prompts_probs'][0]['target_new'])
        cur_sum[f'{data_type}_prob_diff'] = post_prob - pre_prob
        cur_sum[f'{data_type}_pre_prob'] = pre_prob
        cur_sum[f'{data_type}_post_prob'] = post_prob
        erased_prop = (pre_prob - post_prob) / pre_prob
        erased_prop = np.max([0, erased_prop])
        recovered_prop = 1 - (1 - post_prob) / (1 - pre_prob)
        recovered_prop = np.max([2e-8, recovered_prop])
        abs_diff = np.abs(post_prob-pre_prob)
        cur_sum[f'{data_type}_recovered'] = recovered_prop
        cur_sum[f'{data_type}_erased'] = erased_prop
        max_abs_diff = np.abs(pre_prob - .5) + .5
        if data_type != 'neighborhood':
            cur_sum[f'{data_type}_score'] = erased_prop if args.fact_erasure else recovered_prop
        else:
            cur_sum[f'{data_type}_score'] = 1 - abs_diff / max_abs_diff
    try:
        target_score = hmean([
            cur_sum['rewrite_score'], cur_sum['paraphrase_score'], cur_sum['neighborhood_score']
        ])
    except:
        target_score = 0
    cur_sum["target_score"] = target_score
    # compute essence scores 
    if 'essence_score' in data["post"]:
        cur_sum[f"post_essence_ppl"] = data["post"]['essence_score']
        cur_sum[f"pre_essence_ppl"] = data["pre"]['essence_score']
        cur_sum['essence_ppl_diff'] = cur_sum['post_essence_ppl'] - cur_sum['pre_essence_ppl'] # lower is better
    # compute original ROME metrics
    for prefix in ["pre", "post"]:
        # Probability metrics for which new should be lower (better) than true
        for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
            if prefix not in data or key not in data[prefix]:
                continue
            sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
            sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["request_baseline"] > x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["target_new"]) - np.exp(-x["request_baseline"])
                        for x in data[prefix][key]
                    ]
                )
            )
        # Probability metrics for which true should be lower (better) than new
        sum_key_discrete = f"{prefix}_neighborhood_success"
        sum_key_cont = f"{prefix}_neighborhood_diff"
        key = "neighborhood_prompts_probs"
        if prefix in data and key in data[prefix]:
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["request_baseline"] < x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["request_baseline"]) - np.exp(-x["target_new"])
                        for x in data[prefix][key]
                    ]
                )
            )
        # zsRE evaluation metrics
        # for key in ["rewrite", "paraphrase", "neighborhood"]:
        #     sum_key = f"{prefix}_{key}_acc"
        #     key = f"{key}_prompts_correct"
        #     if prefix not in data or key not in data[prefix]:
        #         continue
        #     cur_sum[sum_key].append(np.mean(data[prefix][key]))
        # get harmonic mean averages per point
        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [(
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                (
                    f"{prefix}_rewrite_acc",
                    f"{prefix}_paraphrase_acc",
                    f"{prefix}_neighborhood_acc",
                )]:
                  if k_generalization in cur_sum and k_specificity in cur_sum:
                      cur_sum[f"{prefix}_score"] = hmean([
                                  cur_sum[k_efficacy][0],
                                  cur_sum[k_generalization][0],
                                  cur_sum[k_specificity][0]]
                      )
    # add ROME metrics to record_dict and append to dataframes
    record_dict.update(cur_sum)
    df = pd.DataFrame(record_dict)
    dataframes.append(df)
  if len(dataframes) > 0:
    return_df = pd.concat(dataframes)
  else:
    return_df = pd.DataFrame()
  return return_df


def get_subject_noising_function(model, e_range, hparams, embed_layername):
    # define noise embeddings function
    prng = np.random.RandomState(1) 
    # define function that noises embeddings at tokens_to_mix indices
    def noise_embeddings_f(x, layer):
        # skip noising if seq is a single token (must be bos/eos for open-ended generation)
        if (x.shape[1] == 1):
            return x
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if e_range is not None:
                b, e = e_range
                embeds_noise = torch.from_numpy(prng.randn(x.shape[0], e - b, x.shape[2])).to(x.device)
                x[:, b:e] += hparams.editing_noise * embeds_noise
            # print("added noise to embeds: ", embeds_noise)
            return x
        else:
            return x
    return noise_embeddings_f

def main(
    args,
    alg_name: str,
    model_name: Union[str, Tuple],
    ds_name: str,
    dataset_size_limit: int,
    do_essence_tests: bool,
    skip_generation_tests: bool,
    conserve_memory: bool,
    mt=None,
    verbose=False,
    override_hparams=None,
    overwrite=False,
    correctness_check=False,
    target_prob_check=0,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Get run hyperparameters
    _model_name = model_name.replace('/', '_')
    params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
    if alg_name == 'FT':
      params_path = params_path.replace('.json', '_constr.json')
    hparams = params_class.from_json(params_path)
    args.hparams = hparams
    if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
    print(f"Executing {alg_name} with parameters {hparams}")

    # Determine run directory
    important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
    important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
    exp_name = ROME_experiment_name(args,
                                    model_name.split('/')[-1],
                                    alg_name,
                                    ds_name,
                                    important_hparams)
    run_dir = f'{BASE_DIR}/results/{exp_name}'
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    # copy hparams to results dir
    copy_to_path =  os.path.join(run_dir, 'hparams.json')
    if not os.path.exists(copy_to_path):
        shutil.copyfile(params_path, copy_to_path)
    
    # Instantiate vanilla model
    if mt is None:
      print("Instantiating model")
      model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
      tok = AutoTokenizer.from_pretrained(model_name)
    elif "llava" in model_name:
      model, tok, image_processor = mt.model, mt.tokenizer, mt.image_processor
    else:
      model, tok = mt.model, mt.tokenizer
    tok.pad_token = tok.eos_token

    if args.low_rank:
        low_rank_matrices = ['model.layers.26.mlp.down_proj.weight']
        w = nethook.get_parameter(model, low_rank_matrices[0])
        with torch.no_grad():
            for w_name in low_rank_matrices:
                w = nethook.get_parameter(model, w_name)
                w[...] = do_low_rank(w.detach().clone().float(), 0.1).half()

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR)
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    # Iterate through dataset
    case_ids_exec = []
    for record in tqdm(ds):
        case_id = record["case_id"] if 'case_id' in record else 'known_id'
        case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
        rewrite_this_point = overwrite or not os.path.exists(case_result_path)
         # skip some weird memory issues
        if not(case_id in valid_ids):
            continue
        if case_id==0:
            continue
        # if case_id == 1517 and ((args.alg_name == "ROME" and args.tracing_reversal) or len(hparams.layers) > 1):
        #     continue
        if rewrite_this_point:
            
            print("Starting point: ", case_id)
            # print info for this point
            request = record["requested_rewrite"]
            subject = record["requested_rewrite"]['subject']
            if request['prompt'].count("{}")>1:
               continue
            case_ids_exec.append(case_id)
            prompt = request['prompt'].format(subject)
            request['full_prompt'] = prompt
            target_true = request['target_true']['str']
            paraphrase_prompts = record["paraphrase_prompts"]
            neighborhood_prompts = record["neighborhood_prompts"]
            if verbose:
                print("Updating point:"
                    f" orig update: [{prompt}] -> [{request['target_new']['str']}]"
                    f"\n True label: {target_true}"
                    f"\n Paraphrases: {paraphrase_prompts[:2]}"
                    f"\n Neighbors: {neighborhood_prompts[:2]}")

            
            


            # check if we should skip based on correctness and probability checks
            # if correctness_check or target_prob_check > 0:
            #     is_correct, meets_target_prob = True, True
            #     if correctness_check:
            #         gen_batch = simple_make_inputs(tok, prompts=[prompt])
            #         samples, scores, _ = predict_model(mt, 
            #                                 [prompt], 
            #                                 answers=None, 
            #                                 trigger_phrase=None, 
            #                                 max_decode_steps=36)
            #         is_correct = fewshot_accuracy_sum(samples, [target_true])
            #     if target_prob_check > 0:
            #         preds, scores, _ = predict_model(mt, [prompt], answers=[target_true])
            #         meets_target_prob = scores[0].item() > target_prob_check
            #     if not (is_correct and meets_target_prob):
            #         if verbose:
            #             print(" Skipping this point due to it being incorrect or not meeting the minimum target prob.")
            #             if target_prob_check > 0: 
            #                 print(f" Target prob: {scores[0].item():.4f}")
            #                 print(f" Pred: {preds}")
            #         continue

            # generate essence_texts for evaluation if needed
            if do_essence_tests or not skip_generation_tests:
                essence_prompt = "{} is a".format(subject)
                if verbose:
                    print("GENERATING ESSENCE TEXTS")
                # print(essence_prompt)
                #essence_texts = generate_fast(
                #    model,
                #    tok,
                #    [essence_prompt],
                #    n_gen_per_prompt=5,
                #    max_out_len=100,
                #)
                inputs = tok([essence_prompt], padding=True, return_tensors="pt").to(device)#.cuda()#to(next(model.parameters()).device)
                essence_texts = model.generate(**inputs, max_new_tokens=100, num_return_sequences=5, do_sample=True, top_k=5, cache_position=None)
                essence_texts = list(tok.batch_decode(essence_texts, skip_special_tokens=True))
                # print("essence_texts")
                # print(essence_texts)
                #print(tok.batch_decode(essence_texts.sequences, skip_special_tokens=True))
	        #inputs = tok([essence_prompt], return_tensors="pt")
                #essence_texts = model.generate(**inputs, max_new_tokens=100, num_return_sequences=5)
                # print(essence_texts)
                snips.names_to_samples[subject] = essence_texts
                if verbose:
                    for text in snips.names_to_samples[request['subject']][:2]:
                        print(f" Essence text: {text[:200]}")

            # adjust targets and define 'request_baseline' based on objectives. note model does not necesarily predict 'request_baseline' value before rewriting
            num_noise_samples = 10 if args.fact_forcing and args.alg_name == "FT" else 1 # does not do anything with ROME / MEMIT since these are very low variance
            e_range = find_token_range(tok, substring=subject, prompt_str=prompt)
            request['e_range'] = e_range
            prior_prob = None
            # make noise embeddings_f
            # embed_layername = layername(model, 0, 'embed')
            # noise_embeddings_f = get_subject_noising_function(model, e_range, hparams, embed_layername)
            if args.tracing_reversal:
                gen_batch = simple_make_inputs(tok, prompts=[prompt] * (num_noise_samples))
                with torch.no_grad(), nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings_f) as td:
                    essence_texts = generate_fast(
                        model,
                        tok,
                        [prompt],
                        n_gen_per_prompt=1,
                        max_out_len=12,
                    )
                    new_target = essence_texts[0]
                request['request_baseline'] = request['target_true']['str']
                request['target_new']['str'] = new_target
                request['target_new']['id'] = 'noised-input'
                if verbose:
                    score_batch = make_inputs(tok, [prompt], targets=[new_target])
                    init_target_prob = score_from_batch(model, score_batch)
                    print(f" NEW TARGET PREDICTION: {new_target}")
                    print(f" with init pred prob: {init_target_prob.item():.4f}")
            elif args.fact_erasure:
                # batch = make_inputs(mt.tokenizer, prompts=[prompt] * num_noise_samples, targets=[target_true] * num_noise_samples)
                request['request_baseline'] = mt.tokenizer.eos_token # arbitrary token, won't use these metrics anyway
                request['target_new'] = request['target_true']
       
            elif args.dummy_string:
                # batch = make_inputs(mt.tokenizer, prompts=[prompt] * num_noise_samples, targets=[target_true] * num_noise_samples)
                # request['request_baseline'] = mt.tokenizer.eos_token
                # print(request['target_new']["str"])
                # print(request['target_true']["str"])
                # request['target_new'] = request['target_true']
                request['request_baseline'] = request['target_true']['str']
                request['target_new'] = request['target_empty']
                # print(request)
            elif args.fact_amplification:
                request['request_baseline'] = mt.tokenizer.eos_token # arbitrary token, won't use these metrics anyway
                request['target_new'] = request['target_true']
            elif args.fact_forcing or args.weight_based_tracing:
                gen_batch = simple_make_inputs(tok, prompts=[prompt] * (num_noise_samples))
                _, noised_pred_id = corrupted_forward_pass(mt.model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise)
                noised_pred_token = tok.decode([noised_pred_id])
                request['request_baseline'] = noised_pred_token
                request['target_new'] = request['target_true']
            else:
                request['request_baseline'] = request['target_true']['str']
                # print(request)
                # exit()
            if verbose:
                print(" request baseline: ", request['request_baseline'])

            # language setting
            if args.use_img_token:
                batch = simple_make_inputs(tok, [request['prompt'].format(request['subject'])], image_processor, [request['image_id']], model)
                # input_ids = torch.as_tensor(tokenizer_image_token(sys_prompt.format(request['prompt'].format(request['subject'])), tok, IMAGE_TOKEN_INDEX)).view(1, -1).to(device)
            else:
                input_ids = torch.as_tensor(tok.encode(request['prompt'].format(request['subject']))).view(1, -1).to(device)

            # if args.dummy_string:
            #     target_ids = torch.as_tensor(tok.encode(request['target_true']['str']))[1:].view(1, -1).to(device)
            # else:
            #     target_ids = torch.as_tensor(tok.encode(request['target_new']['str']))[1:].view(1, -1).to(device)
            target_ids = torch.as_tensor(tok.encode(request['target_true']['str']))[1:].view(1, -1).to(device)   

            # image setting    
            if not args.use_img_token:
                img_path = get_image_path(request['image_id'])


                images = load_images(image_files=[img_path])#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
                images_tensor = process_images(images, image_processor,model.config).to(model.device, dtype=torch.float16)
           
           
            # print(images_tensor.shape)
            # images_tensor = preprocess_images(images, return_tensors='pt')['pixel_values'].half().to(device)
            # images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)  

            
                
            # get additional functions and variables based on objectives
            # get hidden representations from corrupted+uncorrupted forward passes to use as targets for weight editing
            if args.weight_based_tracing:
                last_subj_idx = e_range[1]
                with torch.enable_grad():
                    # corrupted forward pass. corrupted_hidden_states will be of shape [n_layers, num_noise_samples, seq_len, hidden_dim]
                    gen_batch = simple_make_inputs(tok, prompts=[prompt] * num_noise_samples)
                    gen_batch['output_hidden_states'] = True
                    _, _, corrupted_hidden_states = corrupted_forward_pass(model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise, output_hidden_states=True)
                    corrupted_hidden_states = torch.stack([corrupted_hidden_states[layer+1] for layer in hparams.layers], dim=0)
                    # clean forward pass
                    gen_batch = simple_make_inputs(tok, prompts=[prompt])
                    clean_hidden_states = model(**gen_batch, output_hidden_states=True, image_size=[336, 336]).hidden_states
                    clean_hidden_states = torch.stack([clean_hidden_states[layer+1] for layer in hparams.layers], dim=0)
                # splice uncorrupted hidden_states into corrupted_hidden_states where they are restored. automatically broadcast across num_noise_samples dimension
                hidden_state_supervision = corrupted_hidden_states
                hidden_state_supervision[:,:,last_subj_idx,:] = clean_hidden_states[:,:,last_subj_idx,:]
            else:
                hidden_state_supervision = None

            # Compute weight changes + record weights that changed
            start = time.time()

            if args.debug:
                # input_ids = torch.as_tensor(tok.encode(sys_prompt.format("Describe the image"))).view(1, -1).to(device)
                # print(tok.batch_decode(model.generate(**batch, do_sample=True, num_return_sequences=5)[:, batch["input_ids"].shape[1]:]))
                # print(tok.batch_decode(target_ids))
                top_k_pre, bottom_k_pre, top_1_pre, bottom_1_pre, target_prob_pre = get_metrics_batch(model, batch, target_ids, args.k, layers_wb_attack)
                print("top_k_pre")
                print(top_k_pre)
                # top_k_pre, bottom_k_pre, top_1_pre, bottom_1_pre, target_prob_pre = get_metrics(model, batch["input_ids"], target_ids, batch["images"], args.k, layers_wb_attack)
                # print(input_ids.shape)
                # input_ids = tok.encode(["Describe the image"])
                # input_ids = torch.as_tensor(tok.encode(sys_prompt.format("Describe the image"))).view(1, -1).to(device)
                # print(tok.batch_decode(model.generate(input_ids, images = images_tensor, do_sample=True, num_return_sequences=5)))
                # print(tok.batch_decode(input_ids))
                # exit()

                # print(top_k_pre, bottom_k_pre, top_1_pre, bottom_1_pre, target_prob_pre)
            



            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            if args.retain_rate:
                sampled = sample(retain_rate_samples, 10)
                # query_inputs, sampled_targets = [x['requested_rewrite']['prompt'].format(x['requested_rewrite']['subject']) for x in sampled], [x['requested_rewrite']['target_true']['str'] for x in sampled]
                # query_inputs, sampled_targets = [DEFAULT_IMAGE_TOKEN + "\n" + "Answer the question in one word:\n Question: {} Answer:".format(x["src"]) for x in sampled], [x["pred"] for x in sampled]
                query_inputs, sampled_targets, image_ids = ["Answer the question in one word:\n Question: {} Answer:".format(x["src"]) for x in sampled], [x["pred"] for x in sampled], [x["image_id"] for x in sampled] 
                # query_inputs, sampled_targets, image_ids = [sys_prompt.format("Describe the image") for x in sampled], [x["pred"] for x in sampled], [x["image_id"] for x in sampled] 
                # print(query_inputs)
                # query_inputs, sampled_targets = ["Describe the image"]*len(query_inputs), [x["pred"] for x in sampled]
                
                batch = simple_make_inputs(tok, query_inputs, image_processor, image_ids, model)
                batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1], image_sizes = batch["image_sizes"][i:i+1]) for i in range(len(batch["input_ids"]))]
                pad_token_id = 0
                outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                preds = tok.batch_decode(outputs, skip_special_tokens=True)
                preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]

                retain_rate_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                preds_preedit = preds

                print(query_inputs)
                print(preds)
             
        
                
                query_inputs, sampled_targets, image_ids = ["Answer the question in one word:\n Question: {} Answer:".format(record['neighborhood_prompts'][i]['prompt'].replace("nq question: ","")) for i in range(len(record['neighborhood_prompts']))], [record['neighborhood_prompts'][i]['target'] for i in range(len(record['neighborhood_prompts']))],  [request['image_id'] for i in range(len(record['neighborhood_prompts']))]
                
                
                batch = simple_make_inputs(tok, query_inputs, image_processor, image_ids, model)
                batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1], image_sizes = batch["image_sizes"][i:i+1]) for i in range(len(batch["input_ids"]))]
                pad_token_id = 0
                outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                preds = tok.batch_decode(outputs, skip_special_tokens=True)
                preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]

                # outputs = model.generate(**batch, do_sample=False, max_new_tokens=36,
                #                   pad_token_id=pad_token_id)
                # outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                # preds = [tok.decode(output) for output in outputs]
                # preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                retain_rate_neighborhood_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                preds_preedit_neighborhood = preds

                if args.neigh_img:
                    targets = [record['alt_ans'][record['neigh_idx'][0]]]
                    print(targets)
                    # targets = record['alt_ans'][:3]
                    # if len(targets) < 3:
                        # targets += ["dummy"]*(3-len(targets))
                    query_inputs, sampled_targets, image_ids, sample_ids = [request['prompt'].format(request['subject'])]*20, targets*20,  [request['image_id']],  [request['id']]
                    retain_rate_img_neighborhood_pre = get_acc_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, sampled_targets, args.img_attack_parap, args.neigh_type, record['neigh_idx'])
                    print("retain_rate_img_neighborhood_pre")
                    print(retain_rate_img_neighborhood_pre)

                    # batch = simple_make_inputs_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, args.img_attack_parap, args.neigh_type)
                    # batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                    # pad_token_id = 0
                    # outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                    #               pad_token_id=0, num_return_sequences=1)[0] for i in range(len(batch["input_ids"]))]
                    # outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                    # preds = tok.batch_decode(outputs, skip_special_tokens=True)
                    # preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                    # print(preds)
                    # print(sampled_targets)
                    # retain_rate_img_neighborhood_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                    # preds_preedit_img_neighborhood = preds
                    # print("retain_rate_img_neighborhood_pre")
                    # print(retain_rate_img_neighborhood_pre)
                    # print(preds_preedit_img_neighborhood)
                    # print(sampled_targets)
                    # print(query_inputs)
                    # print(image_ids)
                    # exit()

                '''
                if args.neigh_mm:
                    targets = record['alt_ans'][:3]
                    if len(targets) < 3:
                        targets += ["dummy"]*(3-len(targets))               
                    prompt_cur = request['prompt'].format(request['subject'])
                    query_inputs, sampled_targets, image_ids, sample_ids = [prompt_cur[:43] + pref + prompt_cur[43:] for pref in prefixes[:3]], targets*3,  [request['image_id']],  [request['id']]
                    retain_rate_mm_neighborhood_pre = get_acc_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, sampled_targets, args.img_attack_parap, args.neigh_type)
                    
                    batch = simple_make_inputs_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, args.img_attack_parap, args.neigh_type)
                    batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                    pad_token_id = 0
                    outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                    outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                    preds = tok.batch_decode(outputs, skip_special_tokens=True)
                    preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                    print(preds)
                    print(sampled_targets)
                    retain_rate_mm_neighborhood_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                    preds_preedit_mm_neighborhood = preds
                    print("retain_rate_mm_neighborhood_pre")
                    print(retain_rate_mm_neighborhood_pre)
                    print(preds_preedit_mm_neighborhood)
                    print(sampled_targets)
                    print(query_inputs)
                    print(image_ids)
                '''
            # if args.debug:
            #     input_ids = torch.as_tensor(tok.encode(request['prompt'].format(request['subject']))).view(1, -1).to(device)
                
            #     if args.dummy_string:
            #         target_ids = torch.as_tensor(tok.encode(" "+request['target_true']['str'])).view(1, -1).to(device)
            #     else:
            #         target_ids = torch.as_tensor(tok.encode(" "+request['target_new']['str'])).view(1, -1).to(device)
            #     top_k, bottom_k, top_1, bottom_1, target_prob = get_metrics_debug(model, input_ids, target_ids, args.k, layers_wb_attack)
            
            # low_rank_matrices = ['model.layers.26.mlp.down_proj.weight']
            # if args.low_rank:
            #     with torch.no_grad():
            #         for w_name in low_rank_matrices:
            #             w = nethook.get_parameter(model, w_name)
            #             w[...] = do_low_rank(w.detach().clone().float(), 0.1)
            # print("required+grad_1")
            # for n, p in model.named_parameters():
            #         if p.requires_grad:
            #             print(n)

            # torch.save(model.state_dict(), "/nas-ssd2/vaidehi/MMMEdit/data/model_preeedit.pt")
            
            with torch.enable_grad(), nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings_f) if args.fact_forcing else nullcontext() as td:
            

              if args.cft_edit:

                edited_model, weights_copy = apply_algo(
                  args,
                  model,
                  tok,
                  image_processor,
                  [request],
                  hparams,
                  copy=False,
                  return_orig_weights=True,
                  num_noise_samples=num_noise_samples,
                  prior_prob=prior_prob,
                  hidden_state_supervision=hidden_state_supervision,
                  **args_conserve_memory,
              )
                # import pdb
                # pdb.set_trace()
                # print("required+grad_2")
                # for n, p in edited_model.named_parameters():
                #     if p.requires_grad:
                #         print(n)

              elif args.lft_edit:
                print("Executing edit method: LORA fine-tuning")
                ques_parap = False
                both_parap = False
                if args.rephrase_defense:
                    ques_parap = True
                if args.dummy_string:
                    defense = "empty_response"
                elif args.fact_erasure:
                    defense = "fact_erasure"
                else:
                    defense = "error_injection"
                lft_data = get_lora_sample_data(request, defense, ques_parap, both_parap)
                print("lft_data")
                print(lft_data)
                # if args.dummy_string:
                #     defense = "empty_response"
                # elif args.fact_erasure:
                #     defense = "fact_erasure"
                # else:
                #     defense = "error_injection"
                # lft_data = get_lora_sample_data(request, defense)
                print(args.edit_vision)
                print(args.edit_mm)
                edited_model, weights_copy = easy_fine_tuning(model, tok, image_processor, "orig", defense, sample_data=lft_data, image_folder=".", learning_rate=args.lora_lr, num_train_epochs=args.epoch, margin_loss=args.margin_loss, entropy_loss=args.entropy_loss, edit_vision=args.edit_vision, edit_mm=args.edit_mm, bf16=False)
                # edited_model, weights_copy = easy_fine_tuning(model, tok, image_processor, defense, sample_data=lft_data, image_folder=".", learning_rate=2e-4, num_train_epochs=5, bf16=False)
                      
            # torch.save(model.state_dict(), "/nas-ssd2/vaidehi/MMMEdit/data/model_postedit.pt")
            # exit()

            


            #   exit()
            # print("required+grad_3")
            # for n, p in edited_model.named_parameters():
            #         if p.requires_grad:
            #             print(n)             
            
            
            if args.ft_after_edit:
                print("Executing post-edit LORA fine-tuning")
                if args.neigh_ft:
                    query_inputs, sampled_targets, image_ids = ["Answer the question in one word:\n Question: {} Answer:".format(record['neighborhood_prompts'][i]['prompt'].replace("nq question: ","")) for i in range(len(record['neighborhood_prompts']))], [record['neighborhood_prompts'][i]['target'] for i in range(len(record['neighborhood_prompts']))],  [request['image_id'] for i in range(len(record['neighborhood_prompts']))]
                else:
                    query_inputs = [x['conversations'][0]['value'] for x in lora_sample_data]
                    image_ids = [x['image'] for x in lora_sample_data]
                    sampled_targets = [x['conversations'][1]['value'] for x in lora_sample_data]
                
                print(get_acc(tok, query_inputs, image_processor, image_ids, model, sampled_targets))

                # defense = "fact_erasure"
                # if args.fact_erasure:
                #     defense = "fact_erasure"

                # else:
                #     if args.dummy_string
                #         defense = "empty_response"
                #     else:
                #         defense = "error_injection"
                # print("required+grad_4")
                # for n, p in edited_model.named_parameters():
                #     if p.requires_grad:
                #         print(n)
                print(lora_sample_data)  
                with torch.enable_grad():
                    edited_model, _ = easy_fine_tuning_pft(edited_model, tok, image_processor, "orig", "error_injection", sample_data=lora_sample_data, image_folder=".", learning_rate=9e-4, num_train_epochs=20, margin_loss=args.margin_loss, entropy_loss=args.entropy_loss, edit_vision=args.edit_vision, edit_mm=True, bf16=False)

                print(get_acc(tok, query_inputs, image_processor, image_ids, model, sampled_targets))
                # print("required+grad_5")                
                # for n, p in edited_model.named_parameters():
                #     if p.requires_grad:
                #         print(n)  
                # for n, p in edited_model.named_parameters():
                #     print(n,p)
                #     assert(p.dtype==torch.float16)
                # exit()
                # edited_model_ft.half()
                        
            
               
            exec_time = time.time() - start
            print("Execution took", exec_time)

            if args.retain_rate:
                
                # query_inputs, sampled_targets = [x['requested_rewrite']['prompt'].format(x['requested_rewrite']['subject']) for x in sampled], [x['requested_rewrite']['target_true']['str'] for x in sampled]
                query_inputs, sampled_targets, image_ids = ["Answer the question in one word:\n Question: {} Answer:".format(x["src"]) for x in sampled], [x["pred"] for x in sampled], [x["image_id"] for x in sampled]

                
                batch = simple_make_inputs(tok, query_inputs, image_processor, image_ids, model)
                # pad_token_id = tok.pad_token_id
                # outputs = edited_model.generate(**batch, do_sample=False, max_new_tokens=36, pad_token_id=pad_token_id)
                # outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                # preds = [tok.decode(output) for output in outputs]
                # preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                # print(batches[0]['input_ids'].dtype)
                # exit()
                batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                pad_token_id = 0
                outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                preds = tok.batch_decode(outputs, skip_special_tokens=True)
                preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]    
                retain_rate = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                actual_retain_rate =  fewshot_accuracy_sum(preds, preds_preedit)/len(preds)
                
        
                
                # query_inputs, sampled_targets = record['neighborhood_prompts'], [record['requested_rewrite']['target_true']['str'] for x in range(len(record['neighborhood_prompts']))]
                # query_inputs, sampled_targets = ["Answer the question in one word:\n Question: {} Answer:".format(record['neighborhood_prompts'][0]['prompt'].replace("nq question: ",""))], [record['requested_rewrite']['target_true']['str']]
                query_inputs, sampled_targets, image_ids = ["Answer the question in one word:\n Question: {} Answer:".format(record['neighborhood_prompts'][i]['prompt'].replace("nq question: ","")) for i in range(len(record['neighborhood_prompts']))], [record['neighborhood_prompts'][i]['target'] for i in range(len(record['neighborhood_prompts']))], [request['image_id'] for i in range(len(record['neighborhood_prompts']))]
                
                
                batch = simple_make_inputs(tok, query_inputs, image_processor, image_ids, model)
                # outputs = edited_model.generate(**batch, do_sample=False, max_new_tokens=36,
                #                   pad_token_id=pad_token_id)
                # outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                # preds = [tok.decode(output) for output in outputs]
                # preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                pad_token_id = 0
                outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                preds = tok.batch_decode(outputs, skip_special_tokens=True)
                preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                retain_rate_neighborhood = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                actual_retain_rate_neighborhood = fewshot_accuracy_sum(preds, preds_preedit_neighborhood)/len(preds)
                
            if args.neigh_img:
                    targets = [record['alt_ans'][record['neigh_idx'][0]]]
                    print(targets)
                    # targets = record['alt_ans'][:3]
                    # if len(targets) < 3:
                        # targets += ["dummy"]*(3-len(targets))
                    query_inputs, sampled_targets, image_ids, sample_ids = [request['prompt'].format(request['subject'])]*20, targets*20,  [request['image_id']],  [request['id']]
                    retain_rate_img_neighborhood_post = get_acc_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, sampled_targets, args.img_attack_parap, args.neigh_type, record['neigh_idx'])
                    print("retain_rate_img_neighborhood_post")
                    print(retain_rate_img_neighborhood_post)
                    # exit()
                    '''
                    targets = record['alt_ans'][:3]
                    if len(targets) < 3:
                        targets += ["dummy"]*(3-len(targets))
                    query_inputs, sampled_targets, image_ids, sample_ids = [request['prompt'].format(request['subject'])]*9, targets*3,  [request['image_id']],  [request['id']]
                    batch = simple_make_inputs_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, args.img_attack_parap, args.neigh_type)

                    batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                    pad_token_id = 0
                    outputs = [edited_model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                    outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                    preds = tok.batch_decode(outputs, skip_special_tokens=True)
                    preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                    print(preds)
                    print(sampled_targets)
                    retain_rate_img_neighborhood_post = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                    print("retain_rate_img_neighborhood_post")
                    print(retain_rate_img_neighborhood_post)
                    # print(preds_preedit_img_neighborhood)
                    print(sampled_targets)
                    print(query_inputs)
                    print(image_ids)
                    exit()
                    '''
            '''
            if args.neigh_mm:
                    targets = record['alt_ans'][:3]
                    if len(targets) < 3:
                        targets += ["dummy"]*(3-len(targets))               
                    prompt_cur = request['prompt'].format(request['subject'])
                    query_inputs, sampled_targets, image_ids, sample_ids = [prompt_cur[:43] + pref + prompt_cur[43:] for pref in prefixes[:3]]*3, targets*3,  [request['image_id']],  [request['id']]
                    batch = simple_make_inputs_neigh(tok, query_inputs, image_processor, image_ids, sample_ids, model, args.img_attack_parap, args.neigh_type)
                    batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
                    pad_token_id = 0
                    outputs = [model.generate(**batches[i], do_sample=False, max_new_tokens=36,
                                  pad_token_id=0)[0] for i in range(len(batch["input_ids"]))]
                    outputs = [list(filter(lambda x: (x != pad_token_id and x!=-200), output)) for output in outputs]
                
                    preds = tok.batch_decode(outputs, skip_special_tokens=True)
                    preds = [pred.replace(sys_prompt_pred.format(query_input), "").strip() for pred, query_input in zip(preds, query_inputs)]
                    print(preds)
                    print(sampled_targets)
                    retain_rate_mm_neighborhood_post = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                    print("retain_rate_mm_neighborhood_post")
                    print(retain_rate_mm_neighborhood_post)
                    print(preds_preedit_mm_neighborhood)
                    print(sampled_targets)
                    print(query_inputs)
                    print(image_ids)
            '''
            # Execute evaluation suite
            start = time.time()
             

            if args.attack == "hp" or "pd":
              
              with torch.no_grad(): 
                # input_ids = torch.as_tensor(tok.encode(request['prompt'].format(request['subject']))).view(1, -1).to(device)
                if args.use_img_token:
                    batch = simple_make_inputs(tok, [request['prompt'].format(request['subject'])], image_processor, [request['image_id']], model)
                    # input_ids = torch.as_tensor(tokenizer_image_token(sys_prompt.format(request['prompt'].format(request['subject'])), tok, IMAGE_TOKEN_INDEX)).view(1, -1).to(device)
                else:
                    input_ids = torch.as_tensor(tok.encode(request['prompt'].format(request['subject']))).view(1, -1).to(device)
             
                # if args.dummy_string:
                #     target_ids = torch.as_tensor(tok.encode(" "+request['target_true']['str']))[1:].view(1, -1).to(device)
                # else:
                #     target_ids = torch.as_tensor(tok.encode(" "+request['target_new']['str']))[1:].view(1, -1).to(device)
                target_ids = torch.as_tensor(tok.encode(request['target_true']['str']))[1:].view(1, -1).to(device)
                # print(request['target_true']['str'])
                # exit()
                # if args.
                # if args.fact_erasure:
                #     target_ids = torch.as_tensor(tok.encode(request['target_true']['str']))[1:].view(1, -1).to(device)
                # else:
                #     target_ids = torch.as_tensor(tok.encode(request['target_new']['str']))[1:].view(1, -1).to(device)
                # print(target_ids)
                

                # input_tok = tok([prompt.format(request["subject"]) for prompt in all_prompts], return_tensors="pt", padding=True,).to("cuda")
                # target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]
                # print(target_ids)
                # print(tok.decode(target_ids[0]))
                # exit()
                top_k, bottom_k, top_1, bottom_1, target_prob = get_metrics_batch(model, batch, target_ids, args.k, layers_wb_attack)                
                print("top_k_post")
                print(top_k)
                # top_k, bottom_k, top_1, bottom_1, target_prob = get_metrics(edited_model, input_ids, target_ids, images_tensor, args.k, layers_wb_attack)
                # if args.debug:
                #     print(top_k, bottom_k, top_1, bottom_1, target_prob)
                #     exit()
            with torch.no_grad(): 

                metrics = {
                    "case_id": case_id,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": ds_eval_method(args, edited_model, tok, image_processor, record, snips, vec, skip_generation_tests, ds_name=ds_name),
                    "topk_metric": list(top_k.to(dtype=torch.float64).cpu().numpy()),
                    "bottomk_metric": list(bottom_k.to(dtype=torch.float64).cpu().numpy()),
                    "top1_metric": list(top_1.to(dtype=torch.float64).cpu().numpy()),
                    "bottom1_metric": list(bottom_1.to(dtype=torch.float64).cpu().numpy()),
                    "target_prob": list(target_prob.to(dtype=torch.float64).cpu().numpy()),
                    "topk_metric_agg": torch.any(top_k==1.0).item(),
                    "bottomk_metric_agg": torch.any(bottom_k==1.0).item(),
                    "topk_or_bottomk_metric_agg": torch.any(torch.tensor([torch.any(top_k==1.0), torch.any(bottom_k==1.0)])).item(),
                    "top1_metric_agg": torch.any(top_1==1.0).item(),
                    "bottom1_metric_agg": torch.any(bottom_1==1.0).item(),
                    "top1_or_bottom1_metric_agg": torch.any(torch.tensor([torch.any(top_1==1.0), torch.any(bottom_1==1.0)])).item(),
                    "target_prob_agg": torch.max(target_prob).item(),
                    "target_prob": list(target_prob.to(dtype=torch.float64).cpu().numpy()),
                    "topk_pre_metric": list(top_k_pre.to(dtype=torch.float64).cpu().numpy()),
                    "bottomk_pre_metric": list(bottom_k_pre.to(dtype=torch.float64).cpu().numpy()),
                    "top1_pre_metric": list(top_1_pre.to(dtype=torch.float64).cpu().numpy()),
                    "bottom1_pre_metric": list(bottom_1_pre.to(dtype=torch.float64).cpu().numpy()),
                    "target_prob_pre": list(target_prob_pre.to(dtype=torch.float64).cpu().numpy()),
                    "topk_pre_metric_agg": torch.any(top_k_pre==1.0).item(),
                    "bottomk_pre_metric_agg": torch.any(bottom_k_pre==1.0).item(),
                    "topk_or_bottomk_pre_metric_agg": torch.any(torch.tensor([torch.any(top_k_pre==1.0), torch.any(bottom_k_pre==1.0)])).item(),
                    "top1_pre_metric_agg": torch.any(top_1_pre==1.0).item(),
                    "bottom1_pre_metric_agg": torch.any(bottom_1_pre==1.0).item(),
                    "top1_or_bottom1_pre_metric_agg": torch.any(torch.tensor([torch.any(top_1_pre==1.0), torch.any(bottom_1_pre==1.0)])).item(),
                    "target_prob_pre_agg": torch.max(target_prob_pre).item(),
                    "target_prob_pre": list(target_prob_pre.to(dtype=torch.float64).cpu().numpy()),
                    "retain_rate": retain_rate,
                    "retain_rate_neighborhood": retain_rate_neighborhood,
                    "retain_rate_pre" : retain_rate_pre,
                    "retain_rate_neighborhood_pre" : retain_rate_neighborhood_pre,
                    "retain_rate_img_neighborhood_pre" : retain_rate_img_neighborhood_pre,
                    "retain_rate_img_neighborhood_post" : retain_rate_img_neighborhood_post,
                    # "retain_rate_mm_neighborhood_pre" : retain_rate_mm_neighborhood_pre,
                    # "retain_rate_mm_neighborhood_post" : retain_rate_mm_neighborhood_post,
                }
                
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
                # import pdb
                # pdb.set_trace()
                # model.load_state_dict(weights_copy)
                metrics["pre"] = ds_eval_method(args, model, tok, image_processor, record, snips, vec, skip_generation_tests, ds_name=ds_name)
                metrics['prior_prob'] = prior_prob
                
                if args.retain_rate:
                    metrics["delta_accuracy"] = metrics["retain_rate_pre"]-metrics["retain_rate"]
                    metrics["delta_accuracy_neighborhood"] = metrics["retain_rate_neighborhood_pre"]-metrics["retain_rate_neighborhood"]
                    metrics["delta_accuracy_img_neighborhood"] = metrics["retain_rate_img_neighborhood_pre"]-metrics["retain_rate_img_neighborhood_post"]
                    # metrics["delta_accuracy_mm_neighborhood"] = metrics["retain_rate_mm_neighborhood_pre"]-metrics["retain_rate_mm_neighborhood_post"]

            print("Evaluation took", time.time() - start)
            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)
            print('\n')
        else:
            if verbose:
              print(f"skipping {case_result_path}, already run")
            else:
              pass

    return case_ids_exec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE", "MEMIT"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--img_attack_parap",
        choices=["orig", "easy", "medium", "hard", "easy_only", "medium_only", "hard_only"],
        default="hard",
        help="type of image paraphrase attacl",
        required=True,
    )
    parser.add_argument(
        "--neigh_type",
        choices=["easy", "medium", "hard", "all", "em"],
        default="all",
        help="type of negh",
        required=True,
    )
    parser.add_argument(
        "--rephrase_defense",
        dest="rephrase_defense",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--model_name",
        choices=["LanguageBind/MoE-LLaVA-Phi2-2.7B-4e", "liuhaotian/llava-v1.6-vicuna-7b", "liuhaotian/llava-v1.5-7b", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf", "huggyllama/llama-7b", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "/playpen-ssd/vaidehi/belief-localization/third_party/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--attack",
        choices=["hp", "pd", "mg", "hess"],
        default="hp",
        help="attack type",
        required=True,
    )
    parser.add_argument(
        "--fact_token",
        choices=["subject_last", "last"],
        default="subject_last",
        help="When using ROME, this is the token index for which to optimize the MLP output vector. last is last in sequence, subject_last is last in subject entity",
        required=False,
    )
    parser.add_argument(
        "--edit_layer",
        type=int,
        default=0,
        help="Layer at which to edit the model weights. Set to -2 to defer to hparam sweep params below",
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre", "mmedit"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--retain_rate",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--ft_after_edit",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--neigh_ft",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--edit_vision",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--edit_mm",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--lft_edit",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--cft_edit",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--lora_lr",
        "-lora_lr",
        type=float,
        default=1e-2,
        help="--lora_lr",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default='1',
        help="Window sizes separated by spaces to use for editing method",
    )
    parser.add_argument(
        "--dataset_size_limit",
        "-n",
        type=int,
        default=1000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--norm_constraint",
        type=float,
        default=-1,
        help="l0 norm constraint on constrained finetuning",
    )
    parser.add_argument(
        "--kl_factor",
        type=float,
        default=-1,
        help="weight on kl div for essence drift regularization",
    )
    parser.add_argument(
        "--v_lr",
        type=float,
        default=-1,
        help="learning rate for finding v* vector in ROME",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous experiment results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="More printing during editing",
    )
    parser.add_argument(
        "--use_img_token",
        action="store_true",
        help="Whether to use the default image token in the prompt",
    )
    parser.add_argument(
        "--entropy_loss",
        dest="entropy_loss",
        action="store_true",
        help="Add entropy loss",
    )
    parser.add_argument(
        "--low_rank",
        dest="low_rank",
        action="store_true",
        help="low_rank",
    )
    parser.add_argument(
        "--entropy_layers",
        dest="entropy_layers",
        nargs='+', type=int, help='layers on which margin loss is to be applied', required=False
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=4,
        help="top-k metric",
    )
    parser.add_argument(
        "--model_parap",
        action="store_true",
        help="model_parap",
    )
    parser.add_argument(
        "--do_essence_tests",
        type=int,
        default=1,
        choices=[0,1],
        help="Do the essence drift generation test regardless of args.skip_generation_tests",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="debug",
    )
    parser.add_argument(
        "--neigh_img",
        dest="neigh_img",
        action="store_true",
        help="neigh_img",
    )
    parser.add_argument(
        "--neigh_mm",
        dest="neigh_mm",
        action="store_true",
        help="neigh_mm",
    )
    parser.add_argument(
        "--dummy_string",
        dest="dummy_string",
        action="store_true",
        help="dummy_string defense",
    )
    parser.add_argument(
        "--margin_loss",
        dest="margin_loss",
        action="store_true",
        help="Add margin loss",
    )
    parser.add_argument(
        "--margin_layers",
        dest="margin_layers",
        nargs='+', type=int, help='layers on which margin loss is to be applied', required=False
    )
    parser.add_argument(
        "--tracing_reversal",
        action="store_true",
        help="Rather than changing output from target_true to target_new, change it to the prediction obtained from the noised causal tracing input",
    )
    parser.add_argument(
        "--fact_forcing",
        action="store_true",
        help="Rather than change o-true to o-new for (s,r,.) input, change o-noise to o-true for (s-noise, r,.) input",
    )
    parser.add_argument(
        "--fact_erasure",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--fact_amplification",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--weight_based_tracing",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--correctness_filter",
        type=int,
        default=0,
        choices=[0,1],
        help="Only eval on points with correct generations or p(target_true) >= .1",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--layers_wb_attack",
        type=str,
        default='1',
        help="layers for attack",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        choices=[0,1],
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.set_defaults(skip_generation_tests=True, conserve_memory=True)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set device
    device = torch.device(f"cuda:{args.gpu}")
    np.random.seed(args.seed)
    torch.cuda.set_device(device)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # experiment checks
    if args.fact_erasure:
        if not("llama" in args.model_name or "llava" in args.model_name):
            pass
            assert args.correctness_filter, "only erase known facts"
    if args.alg_name == "MEMIT":
        assert args.window_sizes != "1", "use window size >=1 with MEMIT"

    # load model
    if args.run:
        torch.set_grad_enabled(False)
        model_name = args.model_name
        torch_dtype = torch.float16 if '20b' in model_name else None
        mem_usage = True
        print("Loading model...")
        if '20b' not in model_name:
            mt = ModelAndTokenizer(model_name, low_cpu_mem_usage=mem_usage, torch_dtype=torch_dtype, cache_dir=MODEL_DIR)
            torch.cuda.empty_cache()
            mt.model.eval().cuda()
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})
        else:
            raise RuntimeError("20b model does not load properly across devices")
            from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", 
                                                        device_map={
                                                            'embed_out' : 0,
                                                            'gpt_neox.embed_in' : 0,
                                                            'gpt_neox.layers': 1,
                                                            'gpt_neox.final_layer_norm' : 0,
                                                        },
                                                        low_cpu_mem_usage=mem_usage,
                                                        torch_dtype=torch_dtype)
            torch.cuda.empty_cache()
            model.eval().cuda()
            tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            mt = ModelAndTokenizer(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})

    # set experiment args
    RUN_EXPERIMENT = args.run # set to false to just collect results
    num_points = args.dataset_size_limit
    alg_name = args.alg_name
    model_name = args.model_name
    assert alg_name in ["FT", "ROME", "MEMIT"]
    hparams_class, _ = ALG_DICT[alg_name]
    ds_name = args.ds_name
    window_sizes = [int(x) for x in args.window_sizes.split()]
    layers_wb_attack = [int(x) for x in args.layers_wb_attack.split()]

    if 'gpt2' in model_name:
        central_layers = list(range(0, 48, 4)) + [17, 47]
        num_layers = 48
    if '6B' in model_name:
        central_layers = list(range(0, 28, 4)) + [5, 27]
        num_layers = 28
        if ds_name == 'zsre':
            central_layers = np.setdiff1d(central_layers, [24])
    if '7b' in model_name:
        central_layers = list(range(0, 32, 4)) #+ [5, 27]
        num_layers = 32
        # if ds_name == 'zsre':
        #     central_layers = np.setdiff1d(central_layers, [24])
    if alg_name == 'FT' and 1 in window_sizes and not args.fact_forcing:
        central_layers = [-1] + central_layers
    if args.edit_layer > -2:
        central_layers = [args.edit_layer]
    print("Starting sweep with hparams:")
    print("- window_sizes: ", window_sizes)
    print("- central_layers: ", central_layers)

    # main experiment loop
    results_dfs = []
    for window_size in window_sizes:
        for central_layer in central_layers:
            override_hparams = get_override_hparams(args, window_size, central_layer, alg_name)
            if RUN_EXPERIMENT:
                case_ids_exec = main(
                    args,
                    alg_name=alg_name,
                    model_name=model_name,
                    ds_name=ds_name,
                    dataset_size_limit=num_points,
                    do_essence_tests=args.do_essence_tests,
                    skip_generation_tests=args.skip_generation_tests,
                    conserve_memory=args.conserve_memory,
                    mt=mt,
                    override_hparams=override_hparams,
                    verbose=args.verbose,
                    overwrite=args.overwrite,
                    correctness_check=args.correctness_filter,
                    target_prob_check=0.0 if args.correctness_filter and args.fact_erasure else 0
                )
            # accumulate results
            exp_name = ROME_experiment_name_from_override_params(args, model_name, alg_name, ds_name, override_hparams, hparams_class)
            editing_results_df = make_editing_results_df(exp_name, num_points, case_ids_exec)
            editing_results_df['edit_method'] = alg_name
            editing_results_df['edit_central_layer'] = central_layer
            editing_results_df['edit_window_size'] = window_size
            results_dfs.append(editing_results_df)
    
    # combine and save results
    results_df = pd.concat(results_dfs)
    _model_name = model_name.split('/')[-1]
    sweep_params = {'ws': window_sizes, 'layers': args.edit_layer}
    ovr_exp_name = sweep_experiment_name(args, _model_name, alg_name, ds_name, sweep_params)
    file_name = f'{ovr_exp_name}.csv'
    save_path = f'{BASE_DIR}/results/{file_name}'
    results_df.to_csv(save_path, index=False)

    print(f"saving csv at {save_path}...")
    print("results shape: ", results_df.shape)

    metrics = ['post_rewrite_success', 'rewrite_prob_diff', 'rewrite_post_prob', 'rewrite_score', 'post_paraphrase_success', 'paraphrase_prob_diff', 'paraphrase_post_prob', 'paraphrase_score', 'post_neighborhood_success', 'neighborhood_prob_diff', 'neighborhood_score', 'essence_ppl_diff', 'topk_metric_agg', 'bottomk_metric_agg', 'topk_or_bottomk_metric_agg', 'top1_metric_agg', 'bottom1_metric_agg', 'top1_or_bottom1_metric_agg', 'target_prob_agg', 'topk_pre_metric_agg', 'bottomk_pre_metric_agg', 'topk_or_bottomk_pre_metric_agg', 'top1_pre_metric_agg', 'bottom1_pre_metric_agg', 'top1_or_bottom1_pre_metric_agg', 'target_prob_pre_agg', 'retain_rate_neighborhood_pre', 'retain_rate_pre', 'retain_rate_neighborhood', 'retain_rate', 'delta_accuracy', 'delta_accuracy_neighborhood', 'delta_accuracy_img_neighborhood', 'delta_accuracy', 'delta_accuracy_neighborhood', 'delta_accuracy_img_neighborhood', 'delta_accuracy_mm_neighborhood', 'retain_rate_neighborhood_pre', 'retain_rate_pre', 'retain_rate_neighborhood', 'retain_rate', 'retain_rate_img_neighborhood_pre', 'retain_rate_img_neighborhood_post', 'retain_rate_mm_neighborhood_pre', 'retain_rate_mm_neighborhood_post']
    # metrics = ["post_rewrite_acc", "pre_rewrite_acc", "post_paraphrase_acc", "pre_paraphrase_acc", "paraphrase_prompts_correct"]
    # another test code review comment
    if len(window_sizes) == 1 and len(central_layers) == 1:
        print("\nfinal metrics: ")
        for metric in metrics:
            if metric in results_df.columns:
                avg_val = np.mean(results_df.loc[:,metric])
                print(f" {metric:.20s}: {avg_val:.3f}")
            else:
                print(f" missing {metric}")

