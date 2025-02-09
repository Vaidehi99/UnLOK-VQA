import torch
import numpy as np
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
from PIL import Image
import os
import json, pickle

sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:"


# Helper functions for rank reduction
def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

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


def get_image_path(image_id):
  img_path = "/nas-ssd2/dataset/coco2017/train2017/{}.jpg".format(str(image_id).zfill(12))
  if not os.path.exists(img_path):
    img_path = "/nas-ssd2/dataset/coco2017/val2017/{}.jpg".format(str(image_id).zfill(12))
  return img_path

def get_image_path_neigh_mmedit(image_id, sample_id, img_attack_parap, neigh_type, neigh_idx):
    img_path_orig = get_image_path(image_id)
    img_path_easy = ["/nas-ssd2/vaidehi/MMMEdit/data/mmedit_new/neigh_easy/{}_{}.jpg".format(sample_id, j) for j in neigh_idx]
    img_path_medium = ["/nas-ssd2/vaidehi/MMMEdit/data/mmedit_new/neigh_med/{}_{}.jpg".format(sample_id, j) for j in neigh_idx]
    img_path_hard = ["/nas-ssd2/vaidehi/nlp13/data/neighborhood_images_hard/mmedit/{}_{}.jpg".format(sample_id, j) for j in neigh_idx]
    if neigh_type=="easy":
      return img_path_easy
    if neigh_type=="medium":
      return img_path_medium
    if neigh_type=="hard":
      return img_path_hard
    if neigh_type=="em":
      return img_path_easy + img_path_medium
    return img_path_easy + img_path_medium + img_path_hard

def get_image_path_neigh(image_id, sample_id, img_attack_parap, neigh_type, neigh_idx):
    img_path_orig = get_image_path(image_id)
    img_path_easy = ["/nas-ssd2/vaidehi/MMMEdit/data/okvqa_new/neigh_easy/{}_{}.jpg".format(sample_id, j) for j in neigh_idx]
    img_path_medium = ["/nas-ssd2/vaidehi/MMMEdit/data/okvqa_new/neigh_med/{}_{}.jpg".format(sample_id, j) for j in neigh_idx]
    img_path_hard = ["/nas-ssd2/vaidehi/nlp13/data/neighborhood_images_hard/okvqa/{}_{}.jpg".format(sample_id, j) for j in neigh_idx]
    if neigh_type=="easy":
      return img_path_easy
    if neigh_type=="medium":
      return img_path_medium
    if neigh_type=="hard":
      return img_path_hard
    if neigh_type=="em":
      return img_path_easy + img_path_medium
    return img_path_easy + img_path_medium + img_path_hard
    # img2id = json.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/img2id.json","r"))
    
    if img_attack_parap=="hard":
      if not os.path.exists(img_path_hard):
        # raise AssertionError
        img_path_hard = img_path_easy
      attack_images = [img_path_orig, img_path_easy, img_path_medium, img_path_hard]
    elif img_attack_parap=="medium":
      attack_images = [img_path_orig, img_path_easy, img_path_medium]
    elif img_attack_parap=="easy":
      if not os.path.exists(img_path_easy):
        raise AssertionError
      attack_images = [img_path_orig, img_path_easy]
    elif img_attack_parap=="hard_only":
      if not os.path.exists(img_path_hard):
        # raise AssertionError
        img_path_hard = img_path_easy
      attack_images = [img_path_hard]
    elif img_attack_parap=="medium_only":
      if not os.path.exists(img_path_medium):
        # print(img_path_medium)
        img_path_medium = img_path_easy
      attack_images = [img_path_medium]
    elif img_attack_parap=="easy_only":
      attack_images = [img_path_easy]
    else:
      attack_images = [img_path_orig]
    return attack_images


def get_image_path_parap(image_id, sample_id, img_attack_parap):
    img_path_orig = get_image_path(image_id)
    img_path_easy = "/nas-ssd2/vaidehi/nlp13/data/parap_images/okvqa/{}.jpg".format(sample_id)
    img_path_medium = "/nas-ssd2/vaidehi/nlp13/data/parap_images_medium/okvqa/{}.jpg".format(sample_id)
    img_path_hard = "/nas-ssd2/vaidehi/nlp13/data/paraphrase_images_hard/okvqa_yolo_dino_bert/processed_images/{}.jpg".format(sample_id)
    # img2id = json.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/img2id.json","r"))
    
    if img_attack_parap=="hard":
      if not os.path.exists(img_path_hard):
        # raise AssertionError
        img_path_hard = img_path_easy
      attack_images = [img_path_orig, img_path_easy, img_path_medium, img_path_hard]
    elif img_attack_parap=="medium":
      attack_images = [img_path_orig, img_path_easy, img_path_medium]
    elif img_attack_parap=="easy":
      if not os.path.exists(img_path_easy):
        raise AssertionError
      attack_images = [img_path_orig, img_path_easy]
    elif img_attack_parap=="hard_only":
      if not os.path.exists(img_path_hard):
        # raise AssertionError
        img_path_hard = img_path_easy
      attack_images = [img_path_hard]
    elif img_attack_parap=="medium_only":
      if not os.path.exists(img_path_medium):
        # print(img_path_medium)
        img_path_medium = img_path_easy
      attack_images = [img_path_medium]
    elif img_attack_parap=="easy_only":
      attack_images = [img_path_easy]
    else:
      attack_images = [img_path_orig]
    return attack_images


def get_image_path_parap_mmedit(image_id, sample_id, img_attack_parap):
    img_path_orig = get_image_path(image_id)
    img_path_easy = "/nas-ssd2/vaidehi/nlp13/data/parap_images/mmedit/{}.jpg".format(sample_id)
    img_path_medium = "/nas-ssd2/vaidehi/nlp13/data/parap_images_medium/mmedit/{}.jpg".format(sample_id)
    img_path_hard = "/nas-ssd2/vaidehi/nlp13/data/paraphrase_images_hard/mmedit/vqa_yolo_dino_bert/processed_images/{}.jpg".format(sample_id)
    # img2id = json.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/img2id.json","r"))
    
    if img_attack_parap=="hard":
      if not os.path.exists(img_path_hard):
        # raise AssertionError
        img_path_hard = img_path_easy
      attack_images = [img_path_orig, img_path_easy, img_path_medium, img_path_hard]
    elif img_attack_parap=="medium":
      attack_images = [img_path_orig, img_path_easy, img_path_medium]
    elif img_attack_parap=="easy":
      if not os.path.exists(img_path_easy):
        raise AssertionError
      attack_images = [img_path_orig, img_path_easy]
    elif img_attack_parap=="hard_only":
      if not os.path.exists(img_path_hard):
        # raise AssertionError
        img_path_hard = img_path_easy
      attack_images = [img_path_hard]
    elif img_attack_parap=="medium_only":
      if not os.path.exists(img_path_medium):
        # print(img_path_medium)
        img_path_medium = img_path_easy
      attack_images = [img_path_medium]
    elif img_attack_parap=="easy_only":
      attack_images = [img_path_easy]
    else:
      attack_images = [img_path_orig]
    # print("attack_images")
    # print(attack_images)
    return attack_images
# def collect_logits(model, input_ids, margin_layers=[46, 47]):
#     if True:
#         out = model(input_ids, output_hidden_states=True).hidden_states
#         # max_layers = len(out)
#         # out = torch.stack([out[i-max_layers] for i in margin_layers], dim=0)
#         lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
#         decoder_out = lens_decoder(out)
        
#     layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
#     return layer_logits

def simple_make_inputs(tokenizer, prompts, image_processor, image_ids, model, device="cuda"):
    # token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    input_lens = len(tokenizer.encode("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"))
    token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    
    # print(tokenizer_image_token("[PAD]", tokenizer))
    # exit()
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 2
    # print(token_lists[0][input_lens-2:input_lens+2])
    # print(token_lists[0])
    # exit()
    # input_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
    input_ids = [[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i] for i in range(len(token_lists))]
   
    # input_ids = token_lists
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    # attention_mask = [[1] * len(t) for t in token_lists]

    img_paths = [get_image_path(img_id) for img_id in image_ids]


    images = load_images(image_files=img_paths)#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
    # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
    print(len(img_paths))
    print("img_paths")
    print(image_ids)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    )
    if isinstance(images_tensor, list):
      print([x.shape for x in images_tensor])
      images_tensor = torch.cat(images_tensor, dim=0).to(model.device, dtype=torch.float16)
    else:
      print(images_tensor.shape)
      images_tensor = images_tensor.to(model.device, dtype=torch.float16)



    print(len(input_ids))
    print(images_tensor.shape)
    assert (images_tensor.shape[0]==len(input_ids))
    image_sizes = [img.size for img in images]

    # images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
        images = images_tensor,
        image_sizes = image_sizes
    )

    # return [dict(
    #     input_ids=torch.tensor(input_ids[i:(i+1)]).to(device),
    #     attention_mask=torch.tensor(attention_mask[i:(i+1)]).to(device),
    #     images = images_tensor[i:(i+1)]
    # ) for i in range(len(input_ids))]


def simple_make_inputs_neigh(tokenizer, prompts, image_processor, image_ids, sample_ids, model, img_attack_parap, neigh_type, neigh_idx, device="cuda"):
    # token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 2
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]

    # img_path = "/nas-ssd2/dataset/coco2017/val2017/{}"
    attack_images = []

    for k in range(len(image_ids)):
      attack_images += get_image_path_neigh(image_ids[k], sample_ids[k], img_attack_parap, neigh_type, neigh_idx)
    # print(attack_images)
    for i in range(len(attack_images)):
      if not(os.path.exists(attack_images[i])):
        attack_images[i] = "/nas-ssd2/vaidehi/MMMEdit/data/okvqa/neighborhood_images_alt_ans_hard/3455_0.jpg"

    print(attack_images)

    images = load_images(image_files=attack_images)#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
    # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
    
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    image_sizes = [img.size for img in images]
    
    # print(torch.tensor(input_ids).shape)
    # if images_tensor.shape[0]==1:
    #   images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)
    # elif len(input_ids)==1:
    #   input_ids = input_ids*images_tensor.shape[0]
    #   attention_mask = attention_mask*images_tensor.shape[0]
    # if True:
    #   num_images = images_tensor.shape[0]
    #   images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
    #   input_ids = [element for i in range(num_images) for element in input_ids]
    #   attention_mask = [element for i in range(num_images) for element in attention_mask]

    # assert(images_tensor.shape[0]==len(input_ids))

    if images_tensor.shape[0]!=len(input_ids):
        num_images = images_tensor.shape[0]
        images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
        input_ids = [element for i in range(num_images) for element in input_ids]
        attention_mask = [element for i in range(num_images) for element in attention_mask]
        image_sizes = [element for i in range(num_images) for element in image_sizes]
       
    
    print(images_tensor.shape)
    print(image_sizes)
    print(torch.tensor(attention_mask).shape)
    print(torch.tensor(input_ids).shape)

    exit()
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
        images = images_tensor,        
        image_sizes = image_sizes

    )


def simple_make_inputs_old(tokenizer, prompts, image_processor, image_id, model, device="cuda"):
    token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    token_lists = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 2
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]

    img_path = "/nas-ssd2/dataset/coco2017/val2017/{}"
    img_path = "/nas-ssd2/dataset/coco2017/train2017/{}.jpg".format(str(image_id).zfill(12))
    if not os.path.exists(img_path):
      img_path = "/nas-ssd2/dataset/coco2017/val2017/{}.jpg".format(str(image_id).zfill(12))

    images = load_images(image_files=[img_path])#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
    # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)

    

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
        images = images_tensor
    )

def simple_make_inputs_image(tokenizer, prompts, image_processor, image_ids, sample_ids, model, img_attack_parap, device="cuda"):
    # token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 2
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]

    # img_path = "/nas-ssd2/dataset/coco2017/val2017/{}"
    attack_images = []

    for k in range(len(image_ids)):
      attack_images += get_image_path_parap(image_ids[k], sample_ids[k], img_attack_parap)

    images = load_images(image_files=attack_images)#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
    # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print(torch.tensor(input_ids).shape)
    # if images_tensor.shape[0]==1:
    #   images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)
    # elif len(input_ids)==1:
    #   input_ids = input_ids*images_tensor.shape[0]
    #   attention_mask = attention_mask*images_tensor.shape[0]
    # if True:
    #   num_images = images_tensor.shape[0]
    #   images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
    #   input_ids = [element for i in range(num_images) for element in input_ids]
    #   attention_mask = [element for i in range(num_images) for element in attention_mask]

    # assert(images_tensor.shape[0]==len(input_ids))

    if images_tensor.shape[0]!=len(input_ids):
        num_images = images_tensor.shape[0]
        images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
        input_ids = [element for i in range(num_images) for element in input_ids]
        attention_mask = [element for i in range(num_images) for element in attention_mask]
       
    

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
        images = images_tensor
    )



def make_inputs(tokenizer, image_processor, prompts, image_ids, sample_ids, model, img_attack_parap, targets=None, device="cuda"):
    # prompts = [sys_prompt.format(p) for p in prompts]

    # make tensor inputs for pytorch model with right-padding
    # i=0
    # print(targets[i])
    # print(tokenizer.encode(targets[i], add_special_tokens=False))
    # print(tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]))
    # print(prompts[i]+tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]))
    # print(len(prompts))
    # print(tokenizer.encode(targets[0], add_special_tokens=False))
    prompts = [prompts[i]+tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]) for i in range(len(prompts))]
    # print(prompts)
    # print(targets)
    
    # token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    # print(token_lists[0])
    # print(len(token_lists[0]))
    # exit()
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = 2
    # input_lens = len(tokenizer.encode("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"))

    # if targets is None:
    #   maxlen = max(len(t) for t in token_lists)
    #   input_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
      
    #   # input_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
    #   attention_mask = [[1]*(input_lens-2) + [0] * (maxlen - len(t)) + [1] * (len(t)-input_lens-2) for t in token_lists]
    #   return dict(
    #       input_ids=torch.tensor(input_ids).to(device),
    #       attention_mask=torch.tensor(attention_mask).to(device),
    #   )
    if targets is not None:
      # print(targets)
      # target_lists = [tokenizer.encode(" " + t) for t in targets]
      target_lists = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
      # print(target_lists)
      # print(tokenizer.decode(target_lists[0][0]))
      
      maxlen = max(len(p) + len(t) for p, t in zip(token_lists, target_lists))
      # print(maxlen)
      # print(len(token_lists[0]))
      # exit()
      combine_lists = [p + t for p, t in zip(token_lists, target_lists)]
      # query_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
      query_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
      input_ids = [t + [pad_id] * (maxlen - len(t)) for t in combine_lists]
      print(len(input_ids))
      # query_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
      # input_ids = [[pad_id] * (maxlen - len(t)) + t for t in combine_lists]
      # input_ids = [combine_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(combine_lists[i])) + combine_lists[i][input_lens-2:] for i in range(len(combine_lists))]
      attention_mask = [[1] * len(t) + [0] * (maxlen - len(t)) for t in combine_lists]
      # attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in combine_lists]

      # attention_mask = [[1]*(input_lens-2) + [0] * (maxlen - len(t)) + [1] * (len(t)-input_lens+2) for t in token_lists]

      target_ids = []
      target_indicators = []


      for input_ids_i, target_ids_i in zip(token_lists, target_lists):
          target_indicators_i = [0]*len(input_ids_i) + [1]*len(target_ids_i) + [0]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_indicators.append(target_indicators_i)
          target_ids_i = [pad_id]*len(input_ids_i) + target_ids_i + [pad_id]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_ids.append(target_ids_i)

      # img_path = "/nas-ssd2/dataset/coco2017/val2017/{}"
      # img_path = "/nas-ssd2/dataset/coco2017/train2017/{}.jpg".format(str(image_id).zfill(12))
      # if not os.path.exists(img_path):
      #   img_path = "/nas-ssd2/dataset/coco2017/val2017/{}.jpg".format(str(image_id).zfill(12))
      # img_path = get_image_path(image_id)
      img_paths = []
      for k in range(len(image_ids)):
        img_paths += get_image_path_parap(image_ids[k], sample_ids[k], img_attack_parap) 

 

      images = load_images(image_files=img_paths) #["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
      # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
      images_tensor = process_images(
        images,
        image_processor,
        model.config
      ).to(model.device, dtype=torch.float16)
      # images_tensor = process_images(
      #   images,
      #   image_processor,
      #   model.config
      # ).to(device, dtype=torch.float16)


      if images_tensor.shape[0]!=len(input_ids):
        num_images = images_tensor.shape[0]
        images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
        input_ids = [element for i in range(num_images) for element in input_ids]
        attention_mask = [element for i in range(num_images) for element in attention_mask]
        query_ids = [element for i in range(num_images) for element in query_ids]
        target_ids = [element for i in range(num_images) for element in target_ids]
        target_indicators = [element for i in range(num_images) for element in target_indicators]


      assert(images_tensor.shape[0]==len(input_ids))
   
      return dict(
          input_ids=torch.tensor(input_ids).to(device),
          query_ids=torch.tensor(query_ids).to(device),
          target_ids=torch.tensor(target_ids).to(device),
          target_indicators=torch.tensor(target_indicators).to(device),
          attention_mask=torch.tensor(attention_mask).to(device),
          images = images_tensor
      )
    
def make_inputs_lora(tokenizer, image_processor, prompts, image_ids, sample_ids, model, img_attack_parap, targets=None, device="cuda"):
    # prompts = [sys_prompt.format(p) for p in prompts]

    # make tensor inputs for pytorch model with right-padding
    # i=0
    # print(targets[i])
    # print(tokenizer.encode(targets[i], add_special_tokens=False))
    # print(tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]))
    # print(prompts[i]+tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]))
    # print(len(prompts))
    # print(tokenizer.encode(targets[0], add_special_tokens=False))
    prompts = [prompts[i]+tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]) for i in range(len(prompts))]
    # print(prompts)
    # print(targets)
    
    # token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    # print(token_lists[0])
    # print(len(token_lists[0]))
    # exit()
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = 2
    # input_lens = len(tokenizer.encode("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"))

    # if targets is None:
    #   maxlen = max(len(t) for t in token_lists)
    #   input_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
      
    #   # input_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
    #   attention_mask = [[1]*(input_lens-2) + [0] * (maxlen - len(t)) + [1] * (len(t)-input_lens-2) for t in token_lists]
    #   return dict(
    #       input_ids=torch.tensor(input_ids).to(device),
    #       attention_mask=torch.tensor(attention_mask).to(device),
    #   )
    if targets is not None:
      # print(targets)
      # target_lists = [tokenizer.encode(" " + t) for t in targets]
      target_lists = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
      # print(target_lists)
      # print(tokenizer.decode(target_lists[0][0]))
      
      maxlen = max(len(p) + len(t) for p, t in zip(token_lists, target_lists))
      # print(maxlen)
      # print(len(token_lists[0]))
      # exit()
      combine_lists = [p + t for p, t in zip(token_lists, target_lists)]
      # query_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
      query_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
      input_ids = [t + [pad_id] * (maxlen - len(t)) for t in combine_lists]
      print(len(input_ids))
      # query_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
      # input_ids = [[pad_id] * (maxlen - len(t)) + t for t in combine_lists]
      # input_ids = [combine_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(combine_lists[i])) + combine_lists[i][input_lens-2:] for i in range(len(combine_lists))]
      attention_mask = [[1] * len(t) + [0] * (maxlen - len(t)) for t in combine_lists]
      # attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in combine_lists]

      # attention_mask = [[1]*(input_lens-2) + [0] * (maxlen - len(t)) + [1] * (len(t)-input_lens+2) for t in token_lists]

      target_ids = []
      target_indicators = []


      for input_ids_i, target_ids_i in zip(token_lists, target_lists):
          target_indicators_i = [0]*len(input_ids_i) + [1]*len(target_ids_i) + [0]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_indicators.append(target_indicators_i)
          target_ids_i = [pad_id]*len(input_ids_i) + target_ids_i + [pad_id]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_ids.append(target_ids_i)

      # img_path = "/nas-ssd2/dataset/coco2017/val2017/{}"
      # img_path = "/nas-ssd2/dataset/coco2017/train2017/{}.jpg".format(str(image_id).zfill(12))
      # if not os.path.exists(img_path):
      #   img_path = "/nas-ssd2/dataset/coco2017/val2017/{}.jpg".format(str(image_id).zfill(12))
      # img_path = get_image_path(image_id)
      img_paths = []
      for k in range(len(image_ids)):
        # for img_attack_parap in ["orig", "easy_only"]:
          img_paths += get_image_path_parap(image_ids[k], sample_ids[k], img_attack_parap) 
      print(img_paths)

 

      images = load_images(image_files=img_paths) #["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
      # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
      images_tensor = process_images(
        images,
        image_processor,
        model.config
      ).to(dtype=torch.float32)
      # images_tensor = process_images(
      #   images,
      #   image_processor,
      #   model.config
      # ).to(device, dtype=torch.float16)


      if images_tensor.shape[0]!=len(input_ids):
        num_images = images_tensor.shape[0]
        images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
        input_ids = [element for i in range(num_images) for element in input_ids]
        attention_mask = [element for i in range(num_images) for element in attention_mask]
        query_ids = [element for i in range(num_images) for element in query_ids]
        target_ids = [element for i in range(num_images) for element in target_ids]
        target_indicators = [element for i in range(num_images) for element in target_indicators]


      assert(images_tensor.shape[0]==len(input_ids))

      # print(torch.tensor(input_ids).shape)
      # print(torch.tensor(target_ids).shape)
      # print(torch.tensor(attention_mask).shape)
      # print(torch.tensor(target_indicators).shape)
      # print(images_tensor.shape)
      # print(target_ids)
      # print(target_indicators)
   
      return dict(
          input_ids=torch.tensor(input_ids),#.to(device),
          query_ids=torch.tensor(query_ids),#.to(device),
          target_ids=torch.tensor(target_ids),#.to(device),
          target_indicators=torch.tensor(target_indicators),#.to(device),
          attention_mask=torch.tensor(attention_mask),#.to(device),
          images = images_tensor
      )
    

def make_inputs_image(tokenizer, image_processor, prompts, image_ids, sample_ids, model, img_attack_parap, targets=None, device="cuda"):
    # make tensor inputs for pytorch model with right-padding
    # i=0
    # print(targets)
    # print(targets[i])
    # print(tokenizer.encode(targets[i], add_special_tokens=False))
    # print(tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]))
    # print(prompts[i]+tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]))
    # print(len(prompts))
    # print(prompts)
    # print(tokenizer.encode(targets[0], add_special_tokens=False))
    prompts = [prompts[i]+tokenizer.decode(tokenizer.encode(targets[i], add_special_tokens=False)[:-1]) for i in range(len(prompts))]
    token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    
    
    # print(prompts)
    # exit()  

    # attention_mask = [[1] * len(t) for t in token_lists]

    
    # images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)
    # token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = 2
    input_lens = len(tokenizer.encode("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"))

    # if targets is None:
    #       token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
    #       maxlen = max(len(t) for t in token_lists)
    #       input_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
    #       attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    #       return dict(
    #       input_ids=torch.tensor(input_ids).to(device),
    #       attention_mask=torch.tensor(attention_mask).to(device),
    #   )
    if targets is not None:
        token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
        target_lists = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
        maxlen = max(len(p) + len(t) for p, t in zip(token_lists, target_lists))
        combine_lists = [p + t for p, t in zip(token_lists, target_lists)]
        query_ids = [ t + [pad_id] * (maxlen - len(t)) for t in token_lists]
        input_ids = [t + [pad_id] * (maxlen - len(t)) for t in combine_lists]
        attention_mask = [[1] * len(t) + [0] * (maxlen - len(t)) for t in combine_lists]
        target_ids = []
        target_indicators = []

        for input_ids_i, target_ids_i in zip(token_lists, target_lists):
          target_indicators_i = [0]*len(input_ids_i) + [1]*len(target_ids_i) + [0]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_indicators.append(target_indicators_i)
          target_ids_i = [pad_id]*len(input_ids_i) + target_ids_i + [pad_id]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_ids.append(target_ids_i)

    # print(token_lists[0][input_lens-2:input_lens+2])
    # print(token_lists[0])
    # exit()
    
    
   
    # input_ids = token_lists
    
      # maxlen = max(len(t) for t in token_lists)
      # input_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
      # attention_mask = [[1] * len(t) + [0] * (maxlen - len(t)) for t in token_lists]
      # return dict(
      #     input_ids=torch.tensor(input_ids).to(device),
      #     attention_mask=torch.tensor(attention_mask).to(device),
      # )
    # if targets is not None:
    #   # print(targets)
    #   target_lists = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    #   # print(target_lists)
    #   # print(tokenizer.decode(target_lists[0][0]))
      
    #   maxlen = max(len(p) + len(t) for p, t in zip(token_lists, target_lists))
    #   combine_lists = [p + t for p, t in zip(token_lists, target_lists)]
    #   query_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
    #   input_ids = [t + [pad_id] * (maxlen - len(t)) for t in combine_lists]
    #   attention_mask = [[1] * len(t) + [0] * (maxlen - len(t)) for t in combine_lists]
    #   target_ids = []
    #   target_indicators = []

    #   for input_ids_i, target_ids_i in zip(token_lists, target_lists):
    #       target_indicators_i = [0]*len(input_ids_i) + [1]*len(target_ids_i) + [0]*(maxlen - len(input_ids_i)-len(target_ids_i))
    #       target_indicators.append(target_indicators_i)
    #       target_ids_i = [pad_id]*len(input_ids_i) + target_ids_i + [pad_id]*(maxlen - len(input_ids_i)-len(target_ids_i))
    #       print(target_ids_i)
    #       print(target_ids)
    #       target_ids.append(target_ids_i)

        assert(len(image_ids)==1)
        img_paths = []
        for k in range(len(image_ids)):
          img_paths += get_image_path_parap(image_ids[k], sample_ids[k], img_attack_parap) 

   

        images = load_images(image_files=img_paths)#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
        # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)
        images_tensor = process_images(
        images,
        image_processor,
        model.config
      ).to(model.device, dtype=torch.float16)
      
        if images_tensor.shape[0]!=len(input_ids):
          num_images = images_tensor.shape[0]
          images_tensor = torch.cat([images_tensor for i in range(len(input_ids))], 0) #images_tensor.expand(net_size, -1, -1, -1)
      
          input_ids = [element for i in range(num_images) for element in input_ids]
          attention_mask = [element for i in range(num_images) for element in attention_mask]
          query_ids = [element for i in range(num_images) for element in query_ids]
          target_ids = [element for i in range(num_images) for element in target_ids]
          target_indicators = [element for i in range(num_images) for element in target_indicators]
          assert (images_tensor.shape[0]==len(input_ids))

      # img_path = "/nas-ssd2/vaidehi/nlp13/data/images/{}.jpg".format(str(image_id).zfill(12))
      # if not os.path.exists(img_path):
      #   raise AssertionError

      # images = load_images(image_files=[img_path])#["/nas-ssd2/dataset/coco2017/train2017/000000357587.jpg"])#"/nas-ssd2/dataset/coco2017/train2017/000000339761.jpg"])#"/nas-ssd2/dataset/coco2017/val2017/000000297147.jpg"])
      # images_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(device)

      # images_tensor = process_images(
      #   images,
      #   image_processor,
      #   model.config
      # ).to(device, dtype=torch.float16)

      # images_tensor = images_tensor.expand(torch.tensor(input_ids).shape[0], -1, -1, -1)
      
        return dict(
          input_ids=torch.tensor(input_ids).to(device),
          query_ids=torch.tensor(query_ids).to(device),
          target_ids=torch.tensor(target_ids).to(device),
          target_indicators=torch.tensor(target_indicators).to(device),
          attention_mask=torch.tensor(attention_mask).to(device),
          images = images_tensor
      )


def pull_prompt_from_data(data, k):
  prompt_idx = np.random.choice(np.arange(len(data)), size=k, replace=False)
  prompt_ex = data.iloc[prompt_idx]
  eval_idx = np.setdiff1d(np.arange(len(data)), prompt_idx)
  eval_data = data.iloc[eval_idx]
  return prompt_ex, eval_data

def score_from_batch(model, batch, return_log_probs=False):
  # print("Running score_from_batch")
  model_batch = {}
  # model_batch['input_ids'] = batch['input_ids'][:,:-1]
  # model_batch['attention_mask'] = batch['attention_mask'][:,:-1]
  # model_batch['target_ids'] = batch['target_ids'][:,1:]
  # model_batch['target_indicators'] = batch['target_indicators'][:,1:]
  # model_batch['images'] = batch['images']
  print(batch['input_ids'].shape)
  model_batch = {
      'input_ids' : batch['input_ids'],
      'attention_mask' : batch['attention_mask'],
      'images': batch['images']
  }

  # pickle.dump(model_batch, open("/nas-ssd2/vaidehi/nlp13/neighborhood/batch1.pkl", "wb"))
  # exit()




  # print(model_batch)

  # pickle.dump(model_batch, open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/util/model_batch.pkl", "wb"))
  
  target_tokens = batch['target_ids']
  target_mask = batch['target_indicators']

  
  
#   target_mask[0, -2] = 0
#   target_mask[0, -3] = 0
  # print(batch['input_ids'])
  # print(batch['input_ids'].shape)
  # print(target_tokens)
  # print(target_mask)
  # exit()  
  # print(model_batch['input_ids'].shape)
 
  logits = model(**model_batch).logits
  # logits = model(**model_batch).logits



  # print(logits)
  # max_indices = torch.topk(logits[0, -1, :], 10).indices
  # print(max_indices)
  # print(model_batch['input_ids'])

  log_probs = torch.log_softmax(logits, dim=-1)
  
  # align probs and target mask by cutting off one token idx from the ends
  log_probs = log_probs[:,:-1,:] # batch_size x seq_len x vocab_size
  # print(log_probs.shape)

  target_tokens = target_tokens[:,1:] # batch_size x seq_len

  target_mask = target_mask[:,1:]
  
  # now iterate over examples and tokens, collecting the target token prob
  
  # print(log_probs.shape)
  log_probs = log_probs[:, -target_tokens.shape[1]:,:]
  # print(log_probs.shape)
  log_probs = torch.gather(log_probs, -1, target_tokens.unsqueeze(-1)).squeeze(-1)
  # print(log_probs.shape)
 
  # will sum up log probs, so zero out log_probs for non-target indices
  log_probs = target_mask * log_probs
  seq_log_probs = log_probs.sum(-1)
  # print("prob")
  # print(torch.exp(seq_log_probs))
  # import pdb
  # pdb.set_trace() 
  if return_log_probs:
    return seq_log_probs
  else:
    return torch.exp(seq_log_probs)


def score_from_batch_new(batch, logits, return_log_probs=False):
  # print("Running score_from_batch")
  # model_batch = {
  #     'input_ids' : batch['input_ids'],
  #     'attention_mask' : batch['attention_mask']
  # }
  # print(model_batch['input_ids'])
  # print(model_batch['attention_mask'])
  # print(batch['target_ids'])
  target_tokens = batch['target_ids']
  # print("print(target_ids)")
  # print(target_tokens)
  # target_mask = batch['target_indicators']
  # print(target_mask)
  # logits = model(**model_batch).logits
  # print("logits")
  # print(logits.shape)
  log_probs = torch.log_softmax(logits, dim=-1)
  # print(log_probs)
  # align probs and target mask by cutting off one token idx from the ends
  log_probs = log_probs[:,:-1,:] # batch_size x seq_len x vocab_size
  # print(log_probs.shape)
  target_tokens = target_tokens[:,1:] # batch_size x seq_len
  # print(target_tokens.shape)
  # target_mask = target_mask[:,1:]
  # print(target_mask.shape)
  # now iterate over examples and tokens, collecting the target token prob
  # print("logits b")
  # # print(log_probs)
  # print(log_probs.shape)
  # print([x for x in target_tokens if x!=50256])
  # print(target_tokens.shape)
  log_probs = torch.gather(log_probs, -1, target_tokens.unsqueeze(-1)).squeeze(-1)
  # print("logits a")
  # # print(log_probs)
  # print(log_probs.shape)
  # will sum up log probs, so zero out log_probs for non-target indices
  # log_probs = target_mask * log_probs
  # print("log_probs")
  # print(log_probs)
  # print(log_probs.shape)
  # exit()
  seq_log_probs = log_probs.sum(-1)
  # print(seq_log_probs.shape)
  # exit()

  if return_log_probs:
    return seq_log_probs
  else:
    return torch.exp(seq_log_probs)

def score_model(mt, query_inputs, targets):
  batch = make_inputs(mt.tokenizer, mt.image_processor, query_inputs, targets)
  return score_from_batch(mt.model, batch)

def predict_model(mt, 
                query_inputs,
                answers=None, 
                trigger_phrase=None,
                max_decode_steps=None,
                score_if_generating=False):
  assert not isinstance(query_inputs, str), "provide queries as list"
  with torch.no_grad():
    generate_and_score = (answers is None)
    batch = make_inputs(mt.tokenizer, mt.image_processor, query_inputs, targets=answers)
    if generate_and_score:
      pad_token_id = mt.tokenizer.pad_token_id
      pad_token_id = pad_token_id if pad_token_id else 0
      outputs = mt.model.generate(**batch, do_sample=False, max_new_tokens=max_decode_steps,
                                  pad_token_id=pad_token_id)
      outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
      preds = [mt.tokenizer.decode(output) for output in outputs]
      preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
      # for some reason huggingface generate not giving generation probs, so we recalculate
      if score_if_generating: 
        batch = make_inputs(mt.tokenizer, query_inputs, targets=preds)
        scores = score_from_batch(mt.model, batch)
      else:
        scores = -100 * np.ones(len(preds))
    else:
      num_answers = len(answers)
      repeated_inputs = []
      repeated_answers = []
      for input in query_inputs:
        for answer in answers:
          repeated_inputs.append(input)
          repeated_answers.append(answer)
      
      batch = make_inputs(mt.tokenizer, repeated_inputs, repeated_answers)
      scores = score_from_batch(mt.model, batch)
      #print(scores)
      #print(repeated_inputs)
      #exit()
      scores = scores.reshape(-1, num_answers)
      pred_ids = [torch.argmax(ex_answer_probs).item() for ex_answer_probs in scores]
      preds = [answers[pred_id] for pred_id in pred_ids]
  return preds, scores, query_inputs

def get_experiment_name(data_name, task_name, k, instructions, cot_reasons, 
                        custom_tag = None):
  instr = 1*(instructions is not None)
  cot = 1*(cot_reasons is not None)
  _custom_tag = f'_{custom_tag}'
  exp_name = f'{data_name}_{task_name}_k{k}_instr{instr}_cot{cot}{custom_tag}'
  return exp_name

def str_clean(input):
  if input is not None:
    return input.strip().lower()
  else:
    return None

def em_accuracy_sum(preds, labels, return_vec=False):
  assert len(preds) == len(labels)
  # strict calculation of accuracy for predictions from fewshot model
  preds = np.array([str_clean(x) for x in preds])
  labels = np.array([str_clean(label) for label in labels])
  correct = (preds==labels)
  if return_vec:
    return correct.sum(), correct
  else:
    return correct.sum()

def fewshot_accuracy_sum(preds, labels, extract_answers=None, return_vec=False):
  # generous calculation of accuracy for predictions from fewshot model
  # an answer is 'predicted' if it appears in the pred str
  # tie breaking is done randomly if the pred str mentions >1 label
  # returns acc sum, optionally the vector of binary 0/1 accs per point
  assert len(preds) == len(labels)
  n_correct = 0
  correct_indicators = []
  # clean arrays
  preds = np.array([str_clean(x) for x in preds])
  labels = np.array([str_clean(label) for label in labels])
  if extract_answers is not None:
    extract_answers = np.array([str_clean(x) for x in extract_answers])
  else:
    extract_answers = []
  # loop through preds and labels
  for pred, label in zip(preds, labels):
    # make label-specific extract_answers as needed
    if label not in extract_answers:
      extract_answers = [label, 'NO_ANSWER_DETECTED']
    answer_to_counts = {answer : 0 for answer in extract_answers}
    # first see if pred is exactly in answers
    if pred in extract_answers:
      answer_to_counts[pred] += 1
    # if not, then count how often labels appear inside of pred
    else:
      for answer in extract_answers:
        if answer in pred:
          answer_to_counts[answer] += 1
    max_count = max(answer_to_counts.values())
    max_preds = [pred for pred in answer_to_counts.keys() if answer_to_counts[pred] == max_count]
    if len(max_preds) == 1:
      use_pred = max_preds[0]
    else:
      use_pred = 'NO_ANSWER_DETECTED'
    correct = (use_pred == label)
    n_correct += correct
    correct_indicators.append(correct)
  if not return_vec:
    return n_correct
  else:
    return n_correct, np.array(correct_indicators)

def first_appearance_fewshot_accuracy_sum(preds, labels, extract_answers, trigger_phrase, return_vec=False):
  # looks for first possible answer appearance after trigger phrase
  # an answer is 'predicted' based on first appearance of an answer choice in the string
  # returns acc sum, optionally the vector of binary 0/1 accs per point
  # note this faces difficulty when answers are subsets of one another
  assert len(preds) == len(labels)
  preds = np.array([str_clean(x) for x in preds])
  extract_answers = [str_clean(answer) for answer in extract_answers]
  n_correct = 0
  correct_indicators = []
  for pred, label in zip(preds, labels):
    answer_positions = {answer : 2e8 for answer in extract_answers}
    pred = str_clean(pred)
    label = str_clean(label)
    trigger_phrase = str_clean(trigger_phrase)
    # extract part of pred after trigger phrase
    if trigger_phrase in pred and trigger_phrase != "":
      pred = pred.split(trigger_phrase)[1]
    else:
      pred = pred
    # take first appearance of an answer in the pred
    # note this faces difficulty when answers are subsets of one another
    for answer in extract_answers:
      if answer in pred:
        answer_positions[answer] = pred.index(answer)
    min_position = min(answer_positions.values())
    earliest_pred = list(filter(lambda tup: tup[1] == min_position, list(answer_positions.items())))
    if len(earliest_pred) == 1:
      use_pred = earliest_pred[0][0]
    else:
      use_pred = 'NA'
    correct = (use_pred == label)
    n_correct += correct
    correct_indicators.append(correct)
  if not return_vec:
    return n_correct
  else:
    return n_correct, np.array(correct_indicators)

def compute_prop_invalid_preds(preds, answers):
  if answers is None or isinstance(answers, np.ndarray):
    return -1
  n_invalid = 0
  for pred in preds:
    none_present = True
    for answer in answers:
      if answer in str_clean(pred):
        none_present=False
    n_invalid += none_present
  return n_invalid / len(preds)

def verbalize(label, answers, inverted_labels=False):
  '''
  maps integer labels to string answers for scoring by LM
  '''
  assert label < len(answers), f"requesting label {label} but only {len(answers)} answers"
  if not inverted_labels:
    return_answer = answers[label]
  else:
    assert len(answers) == 2, "using inverted_labels=True but more than two answers provided"
    return answers[1-label]
  return answers[label] 

def format_example(input, label_str="", answers=None, cot_reason=None, template_id=0):
  if template_id == 0:
    return f"\"{input}\" is {label_str}"
  if template_id == 1:
    return f"The scenario \"{input}\" describes an action that is {label_str}"
  if template_id == 2:
    return f"scenario: \"{input}\" label ({answers[0]}/{answers[1]}): {label_str}"
  # Hendrycks cm format. label is "wrong" or "not wrong". used with k=32
  if template_id == 3: 
    if label_str=="": # assume test input, don't add line breaks at end
      return "{}\nReaction: this is {}".format(input, label_str)
    else:
      return "{}\nReaction: this is {}\n".format(input, label_str)
  if template_id == 4:
    if label_str=="": # assume test input
      return f"\"{input}\""
    else:
      return f"\"{input}\" {cot_reason} Therefore, the action is {label_str}"
  if template_id == 5:
    if label_str=="": # assume test input
      return f"\"{input}\" The action is"
    else:
      return f"\"{input}\" The action is {label_str} because {cot_reason}"
  # control condition for CoT above, but for multiple choice
  if template_id == 6:
    if cot_reason is not None:
      return_str = f"\"{input}\" {cot_reason} Therefore, the action is"
    else:
      return_str = f"\"{input}\" Therefore, the action is"
    if label_str != "": # not a test input
      return_str += f" {label_str}"
    return return_str
  # used with chain of thought reasons that re-specify the action
  if template_id == 7:
    if label_str=="": # assume test input
      return f"\"{input}\""
    else:
      return f"\"{input}\" {cot_reason} {label_str}"
  if template_id == 8: # for factual data completions
    if label_str=="": # assume test input
      return f"{input}"
    else:
      return f"{input} {label_str}"
  else:
    raise ValueError(f"Not implemented template for template_id {template_id}")

def format_prompt(examples, test_input, instructions=None, separator='\n'):
  # takes list of examples, test_input, already processed by format_example
  if len(examples) > 0:
    examples = separator.join(examples)
    prompt = examples + separator + test_input
  else:
    prompt = test_input
  if instructions:
    prompt = instructions + separator + prompt
  return prompt

def format_example_from_df_row(df_row, template_id=0):
  input = df_row.input
  label_str = df_row.label_str
  example = format_example(input, label_str, template_id=template_id)
  return example

def format_prompt_from_df(df, test_input, answers=None, instructions=None, cot_reasons=None, separator='\n', template_id=0, idx=None):
  # read data from df and pass to format_prompt()
  # add chain-of-thought reasons via format_example here
  examples = []
  select_df = df.iloc[idx,:] if idx else df
  for data_num, (_, df_row) in enumerate(select_df.iterrows()):
    input = df_row['input']
    label_str = df_row['label_str']
    cot_reason = cot_reasons[data_num] if cot_reasons else None
    example = format_example(input, label_str, answers=answers, cot_reason=cot_reason, template_id=template_id)
    examples.append(example)
  formatted_test_input = format_example(test_input, template_id=template_id)
  prompt = format_prompt(examples, formatted_test_input, instructions=instructions, separator=separator)
  return prompt

# main eval loop
def fewshot_eval_model(experiment_name, task_name, mt, eval_data, batch_size, 
                       k=0, random_seed=0, n=None, prompt_data=None, 
                       instructions=None, answers=None, template_id=0, cot_reasons=None,
                       max_decode_steps=128, extract_answers=None,
                       trigger_phrase=None,
                       print_examples=0, print_all_wrong=False):
  """Evaluates prediction service model in fewshot manner
  - answers: constraints model outputs to belong in strings in answers
  - extract_answers: str answers to look for in the generated textual output (when answers is none)
  """
  # argument checks
  if k > 0 and prompt_data is None: 
    assert len(prompt_data) >= 1, f"need to provide prompt data of at least len {k}"
  # define stats
  n_correct = 0
  n_str_em = 0
  n_datapoints = 0
  all_preds = []
  all_labels = []
  # task specific info
  task_name_to_hendrycks_em_group_by = {
      'commonsense': 1,
      'deontology': 4,
      'justice': 4,
      'utilitarianism': 1,
      'virtue': 1, # we treat as multiple choice
      'trolley' : 1,
      'factual' : 1,
      'counterfact' : 1,
  }
  if 'virtue' in task_name:
    assert answers is None, "do not use answers with virtue subset"
  if answers and not extract_answers:
    extract_answers = answers
  # subsample eval data if requested
  if n is not None:
    eval_data_loop = eval_data.sample(n=n, random_state=random_seed, replace=False)
  else:
    eval_data_loop = eval_data
  # begin eval loop
  # calculate query batch size based on if len(inputs) * len(answers) can fit in BATCH_SIZE query to model
  effective_batch_size = batch_size if not answers else batch_size // len(extract_answers)
  n_chunks = np.ceil(len(eval_data_loop) / effective_batch_size)
  for batch_num, batch in enumerate(np.array_split(eval_data_loop, n_chunks)):
    if batch_num > 0:
      running_acc = n_correct / n_datapoints 
      check_answers = extract_answers if answers is None else answers
      prop_invalid_preds = compute_prop_invalid_preds(all_preds, check_answers)
      start = '\r' # '\n' if batch_num < 3 else 
      print(f"{start}Batch {batch_num-1} | Acc: {100*running_acc:.2f} | Invalid: {100*prop_invalid_preds:.2f}", end="")
    # make inputs and labels:
    query_inputs = []
    for test_input in batch.input:
      query_input = format_prompt_from_df(prompt_data, test_input, answers=answers, instructions=instructions, cot_reasons=cot_reasons, separator='\n', template_id=template_id)
      query_inputs.append(query_input)
    labels = batch.label_str
    # make multiple choice answers for virtue
    if 'virtue' in task_name:
      answers = []
      for answer_list in batch.answers:
        answers.append(answer_list.split(','))
      answers = np.array(answers)
    # query model. query inputs may be editing when doing chain_of_thought multiple choice
    with torch.no_grad():
      preds, scores, query_inputs = predict_model(mt, 
                                                  query_inputs, 
                                                  answers, 
                                                  trigger_phrase=trigger_phrase, 
                                                  max_decode_steps=max_decode_steps)
    # record stats
    # first case is when we are generating predictions and extracting answers from them
    if answers is None and extract_answers is not None:
      batch_n_correct, correct_vec = first_appearance_fewshot_accuracy_sum(preds, labels, 
                                                                           extract_answers=extract_answers, 
                                                                           trigger_phrase=trigger_phrase,
                                                                           return_vec=True)
    else:
      batch_n_correct, correct_vec = fewshot_accuracy_sum(preds, labels, return_vec=True)
    n_correct += batch_n_correct
    n_str_em += em_accuracy_sum(preds, labels)
    n_datapoints += len(batch)
    all_preds.extend(list(preds))
    all_labels.extend(list(labels))
    if (print_examples>0 and batch_num == 0):
      print_idx = np.arange(min(print_examples, len(batch)))
    elif print_all_wrong:
      print_idx = np.argwhere(1-correct_vec).reshape(-1)
    else:
      print_idx = np.array([])
    if len(print_idx) > 0:
      print(f"\nExamples from batch {batch_num}...")
      print("--------")
      for i in print_idx:
        print(f"Example {i}")
        print(f"point: \n{batch.input.iloc[i]}")
        print(f"prompt: \n{query_inputs[i]}")
        print("pred: ", preds[i])
        print("label: ", labels.iloc[i])
        if isinstance(answers, np.ndarray):
          print("anwers: ", answers[i])
        print("exact scores: ", scores[i])
        print("correct: ", correct_vec[i])
        if 'completion' in batch.columns:
          print("gpt completion: ", batch.completion.iloc[i])
        print("--------")
      print(f"Examples acc: {correct_vec[print_idx].mean():.2f}")
      print("--------\n")
    del batch, preds, labels, scores
  # calculate EM from Hendrycks et al paper
  group_by = task_name_to_hendrycks_em_group_by[task_name]
  hendrycks_em = get_hendrycks_em(all_preds, all_labels, answers, group_by)
  # make df with results
  results_dict = {
      'exp_name' : experiment_name,
      'task_name' : task_name,
      'k' : k,
      'n' : n,
      'seed' : random_seed,
      'acc' : n_correct / n_datapoints,
      'acc_em' : n_str_em / n_datapoints,
      'hendrycks_em': hendrycks_em,
      'prop_invalid': compute_prop_invalid_preds(all_preds, answers)
  }
  results = pd.DataFrame.from_dict({k : [v] for k,v in results_dict.items()})
  print("\nRESULTS:")
  for k,v in results_dict.items():
    if any([x in k for x in ['acc', 'em', 'prop']]):
      v = f"{100*v:.2f}"
    print(f"  {k}: {str(v):10s}")
  return results 
