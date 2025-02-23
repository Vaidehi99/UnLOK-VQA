import sys
sys.path.insert(0,"/nas-ssd2/vaidehi/projects/LLaVA/cache/")
#sys.path.insert(0,"/nas-ssd2/vaidehi/projects/LLaVA/")
import os
os.environ['TRANSFORMERS_CACHE'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
os.environ['HF_HOME'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
from serve.cli import main as run_llava
import pandas as pd
from tqdm import tqdm
import argparse

class args:
    model_path = "liuhaotian/llava-v1.5-13b"#"liuhaotian/llava-v1.5-7b" #"liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3" #"liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
    image_file = "/nas-ssd2/vaidehi/projects/WikiWeb2M/images-filter/{}"
    load_8bit = False
    model_base = None #"lmsys/vicuna-13b-v1.3"
    device = "cuda"
    conv_mode = None
    temperature = 0.2
    max_new_tokens = 512
    load_4bit = True
    image_aspect_ratio = 'pad'
    # txt = "Combine the following text with the image content coherently and summarize including content from both the text and the image and compress to length less than input text: {}"
    txt = "Combine the following text with the image content coherently and summarize including content from both the text and the image and compress to a minimal length that captures most salient information from both modalities: {}"
    debug = True
    data_path = "/nas-ssd2/vaidehi/projects/WikiWeb2M/data_test_valid_urls_desc_summ_unieval_scores.csv"
    img_dir = "/nas-ssd2/vaidehi/projects/WikiWeb2M/images-filter/"
    j = 0
    size = 10000


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
