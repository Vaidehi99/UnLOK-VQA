import sys
# sys.path.insert(0,"/nas-ssd2/vaidehi/projects/LLaVA/cache/")
#sys.path.insert(0,"/nas-ssd2/vaidehi/projects/LLaVA/")
import os
os.environ['TRANSFORMERS_CACHE'] = '/nas-ssd/vaidehi/hf_cache/'
os.environ['HF_HOME'] = '/nas-ssd/vaidehi/hf_cache/'
# from serve.cli import main as run_llava
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import requests
from PIL import Image, ImageFile
from io import BytesIO
from transformers import TextStreamer
from transformers import set_seed
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import requests
from llava.model.builder import load_pretrained_model

# from llava.utils import disable_torch_init



def run_llava(args):
    j = args.j
    size = args.size
    # disable_torch_init()
    existing_images = os.listdir(args.img_dir)
    print(len(existing_images))
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_path, device=args.device)

    processor = LlavaNextProcessor.from_pretrained(args.model_base)
    # model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path, low_cpu_mem_usage=True).to("cuda:0")
#, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    # if args.save:
    # 	model.save_pretrained(args.save_path)
    # 	tokenizer.save_pretrained(args.save_path)

    #setting for generation
    # tokenizer.padding_side = "left" 
    # tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    data = pd.read_csv(args.data_path.format(args.split))
    data = data[data.img_name.isin(existing_images)]
    print(len(data))
    data = data[j*size:(j+1)*size]
    
    summaries = []
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        prompt = args.txt.format(row["txt"])
        image = Image.open(args.image_file.format(row["img_name"]))
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        # print(inputs)
        # print(inputs.keys())
        # print(model)
        if True:
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
                # print(inputs.input_ids.shape)
            # print(output_ids)
            outputs = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(outputs)

            summaries += [outputs]
        # except:
        #     summaries += [""]

    data["llava_v1.6_7b_round1"] = summaries
    data.to_csv(args.out_path.format(args.split, j))




class args:
    model_path = "/nas-ssd/vaidehi/LLaVA_v1.6_data/llava_v1.6_mixtral_hf_lora_12_2.5e-4/"#"/nas-ssd/vaidehi/LLaVA_v1.6_data/ckpt_lora_llava/" #"liuhaotian/llava-v1.5-13b"#"liuhaotian/llava-v1.5-7b" #"liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3" #"liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
    image_file = "/nas-ssd2/vaidehi/projects/WikiWeb2M/images-filter/{}"
    load_8bit = False
    model_base = "llava-hf/llava-v1.6-mistral-7b-hf" #"lmsys/vicuna-13b-v1.3"
    device = "cuda"
    conv_mode = None
    temperature = 0.2
    max_new_tokens = 512
    load_4bit = True
    image_aspect_ratio = 'pad'
    split = "val"
    # txt = "Combine the following text with the image content coherently and summarize including content from both the text and the image and compress to length less than input text: {}"
    # txt = "[INST] <image>\nCombine the following text with the image content coherently and summarize including content from both the text and the image and compress to a minimal length of less than 3 sentences such that it captures most salient information from both modalities: {}[/INST]"
    txt = "[INST] <image>\nCompose a multimodal summary of less than three sentences combining information from both the following text and the image coherently without without attributing the information to text or image in less than three sentences: {} [/INST]"
    debug = True
    data_path = "/nas-ssd/vaidehi/LLaVA_v1.6_data/{}_llava_v1.6_7b_zs_summ_all.csv"
    # data_path = "/nas-ssd2/vaidehi/projects/WikiWeb2M/{}_zeroshot_llava_v1.5_13b_all_filt.csv"
    out_path = "/nas-ssd/vaidehi/LLaVA_v1.6_data/llava_v1.6_7b_round1_summ_{}_{}.csv"
    img_dir = "/nas-ssd2/vaidehi/projects/WikiWeb2M/images-filter/"
    j = 0
    size = 10 #10000


# data = pd.read_csv(args.data_path)
# print(data["img_name"].head())
# exit()

# summaries = []
# for i, row in data.iterrows():
#     args.txt = row["txt"]
#     args.image_file = args.image_file.format(row["img_name"])
#     print(args.image_file)
#     if(i==2):
#         break
#     summaries += [main(args)]
#     if(i>=10):
#         break
# print(summaries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--j", type=int, default=0)
    args.j = parser.parse_args().j
    
    run_llava(args)
