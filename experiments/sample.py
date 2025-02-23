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
sys_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:"

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device="cuda",
)

prompts = ["What's the content of the image?", "Answer the question in one word: What's in the image? Answer:"]
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# image = Image.open("images/australia.jpeg")


# args = type('Args', (), {
#     "model_path": model_path,
#     "model_base": None,
#     "model_name": get_model_name_from_path(model_path),
#     "query": prompt,
#     "conv_mode": None,
#     "image_file": "images/australia.jpeg",
#     "sep": ",",
#     "temperature": 0,
#     "top_p": None,
#     "num_beams": 1,
#     "max_new_tokens": 512
# })()

# eval_model(args)

# exit()

# image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
# if IMAGE_PLACEHOLDER in prompt:
#     if model.config.mm_use_im_start_end:
#         prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
#     else:
#         prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
# else:
#     if model.config.mm_use_im_start_end:
#         prompt = image_token_se + "\n" + prompt
#     else:
#         prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        
# if "llama-2" in model_name.lower():
#     conv_mode = "llava_llama_2"
# elif "v1" in model_name.lower():
#     conv_mode = "llava_v1"
# elif "mpt" in model_name.lower():
#     conv_mode = "mpt"
# else:
#     conv_mode = "llava_v0"
    
# print(conv_mode)

# conv = conv_templates[conv_mode].copy()
# conv.append_message(conv.roles[0], prompt)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()

# prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
prompts = [sys_prompt.format(prompt) for prompt in prompts]
print(prompts)

# input_ids = tokenizer(
#     prompt,
#     return_tensors="pt",
#     padding="longest",
#     max_length=128,
#     truncation=True,
# ).input_ids

prompts = ["A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word:\n Question: What type of object is this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word:\n Question: What type of object is this? Answer: ASSISTANT:"]
print(len(prompts))
prompts = prompts[:1]
token_lists = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]


'''
prompts = ["A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word\n Question: Which breed of dog it this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word:\n Question: What type of object is this? Answer: ASSISTANT:", "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAnswer the question in one word:\n Question: What type of object is this? Answer: ASSISTANT:"]
print(len(prompts))
prompts = prompts[:1]
token_lists = [tokenizer_image_token(sys_prompt.format(p), tokenizer, IMAGE_TOKEN_INDEX) for p in prompts]
if True:    
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = 0
    print(pad_id)
targets = ['Lab', '</s>', 'Lab', '</s>', 'Lab', '</s>']
if targets is not None:
      # print(targets)
      target_lists = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
      # print(target_lists)
      # print(tokenizer.decode(target_lists[0][0]))
      
      maxlen = max(len(p) + len(t) for p, t in zip(token_lists, target_lists))
      combine_lists = [p + t for p, t in zip(token_lists, target_lists)]
      # query_ids = [token_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(token_lists[i])) + token_lists[i][input_lens-2:] for i in range(len(token_lists))]
      query_ids = [t + [pad_id] * (maxlen - len(t)) for t in token_lists]
      input_ids = [t + [pad_id] * (maxlen - len(t)) for t in combine_lists]
      # query_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
      # input_ids = [[pad_id] * (maxlen - len(t)) + t for t in combine_lists]
      # input_ids = [combine_lists[i][:input_lens-2]+[pad_id] * (maxlen - len(combine_lists[i])) + combine_lists[i][input_lens-2:] for i in range(len(combine_lists))]
      attention_mask = [[1] * len(t) + [0] * (maxlen - len(t)) for t in combine_lists]
      # attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in combine_lists]

      # attention_mask = [[1]*(input_lens-2) + [0] * (maxlen - len(t)) + [1] * (len(t)-input_lens+2) for t in token_lists]

      target_ids = []
      target_indicators = []

      print(input_ids)
      print(attention_mask)

      for input_ids_i, target_ids_i in zip(token_lists, target_lists):
          target_indicators_i = [0]*len(input_ids_i) + [1]*len(target_ids_i) + [0]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_indicators.append(target_indicators_i)
          target_ids_i = [pad_id]*len(input_ids_i) + target_ids_i + [pad_id]*(maxlen - len(input_ids_i)-len(target_ids_i))
          target_ids.append(target_ids_i)

input_ids = torch.tensor(input_ids)

# input_ids_0= tokenizer_image_token(
#     prompts[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
# print(input_ids_0)
# input_ids_1 = tokenizer_image_token(
#     prompts[1], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
# print(input_ids_1)
# input_ids = torch.tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
#          21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
#            322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
#          29889,  3148,  1001, 29901, 29871,  -200,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0, 29871,    13,
#          22550,   278,  1139,   297,   697,  1734, 29901,    13,   894, 29901,
#           1724,  1134,   310,  4402,   338,   445, 29973,   673, 29901,   319,
#           1799,  9047, 13566, 29901],
#         [    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
#          21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
#            322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
#          29889,  3148,  1001, 29901, 29871,  -200,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0, 29871,    13, 22550,
#            278,  1139,   297,   697,  1734, 29901,    13,   894, 29901,  6804,
#            723,   366, 12234,  1284,   445, 13019, 29973,   673, 29901,   319,
#           1799,  9047, 13566, 29901]]).to(model.device)
# print(input_ids)
img = process_images(
    [image]*6,
    image_processor,
    model.config
).to(model.device, dtype=torch.float16)
print(img.shape)
print(input_ids.shape)
input_ids = input_ids[0:1]
img = img[0:1]
# img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
# img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
#     images_tensor = process_images(
# print(input_ids.shape)

# targets = input_ids.clone()

'''

import pickle
batch = pickle.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/util/model_batch.pkl", "rb"))
print(batch)
batches = [dict(input_ids=batch['input_ids'][i:i+1], attention_mask=batch["attention_mask"][i:i+1], images = batch["images"][i:i+1]) for i in range(len(batch["input_ids"]))]
# logits = model(**batches[0]).logits
# print(logits.shape)
print(len(batches[0]["input_ids"][0]))
print(len(token_lists[0]))


input_ids_0= tokenizer_image_token(
    prompts[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
print(input_ids_0)
print(torch.tensor(token_lists))
print(batches[0]["input_ids"])
# hidden_states = model(
#    **batches[0],
#     output_hidden_states=True
#     # max_length=30,
# ).hidden_states

# hidden_states = model(
#    input_ids=torch.tensor(token_lists).to(model.device),
#    images = batches[0]["images"],
#     output_hidden_states=True
#     # max_length=30,
# ).hidden_states


hidden_states = model(
    input_ids=input_ids_0,
    images=batches[0]["images"],
    output_hidden_states=True
    # max_length=30,
).hidden_states

hidden_states = model(
   input_ids=torch.tensor(token_lists).to(model.device),
   images = batches[0]["images"],
    output_hidden_states=True
    # max_length=30,
).hidden_states

hidden_states = model(
  input_ids=batches[0]["input_ids"],
  images = batches[0]["images"],
    output_hidden_states=True
    # max_length=30,
).hidden_states

# print(len(hidden_states))
# print(hidden_states[0].shape)
# exit()
print(input_ids_0)
print(batches[0]["input_ids"])
print(batches[0]["input_ids"].shape)
print(model(input_ids=input_ids_0,
    images=batches[0]["images"]).logits.shape)


generate_ids = model.generate(
    input_ids=batches[0]["input_ids"],
    images=batches[0]["images"],
    max_length=30,
    max_new_tokens=512,
)
# print(generate_ids.shape)
# exit()
input_token_len = batches[0]["input_ids"].shape[1]
n_diff_input_output = (batches[0]["input_ids"] != generate_ids[:, :input_token_len]).sum().item()
if n_diff_input_output > 0:
    print(
        f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
    )
outputs = tokenizer.batch_decode(
    generate_ids[:, input_token_len:], skip_special_tokens=True
)
print("1")
output0 = outputs[0].strip()
print(output0)

exit()
print("2")
output1 = outputs[1].strip()
print(output1)
print("3")
print(tokenizer.decode([29871,    13,
         22550,   278,  1139,   297,   697,  1734, 29901,    13,   894, 29901,
          3750,   338,   393, 11203, 23407, 29973,   673, 29901,   319,  1799,
          9047, 13566, 29901]))

print("4")
print(tokenizer.decode([29871,    13, 22550,   278,
          1139,   297,   697,  1734, 29901,  1724, 29915, 29879,   297,   278,
          1967, 29973,   673, 29901,   319,  1799,  9047, 13566, 29901]))