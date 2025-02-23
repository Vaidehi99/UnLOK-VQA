import sys
sys.path.append("/nas-ssd/vaidehi/LLaVA_v1.6_data/LLaVA_may/")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from io import BytesIO
from transformers import TextStreamer
from transformers import set_seed
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TRANSFORMERS_CACHE'] = '/nas-ssd/vaidehi/hf_cache/'
os.environ['HF_HOME'] = '/nas-ssd/vaidehi/hf_cache/'
import sys
# sys.path.insert(0,"/nas-ssd2/vaidehi/projects/LLaVA/cache/")
import os
os.environ['TRANSFORMERS_CACHE'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import pandas as pd
from tqdm import tqdm


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model_path = "liuhaotian/llava-v1.6-mistral-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

# args = type('Args', (), {
#     "model_path": model_path,
#     "model_base": None,
#     "model_name": get_model_name_from_path(model_path),
#     "query": prompt,
#     "conv_mode": None,
#     "image_file": image_file,
#     "sep": ",",
#     "temperature": 0,
#     "top_p": None,
#     "num_beams": 1,
#     "max_new_tokens": 512
# })()

class args:
    model_path = "liuhaotian/llava-v1.6-mistral-7b" #"llava-hf/llava-v1.6-mistral-7b-hf" #"liuhaotian/llava-v1.6-mistral-7b" #"liuhaotian/llava-v1.5-13b"#"liuhaotian/llava-v1.5-7b" #"liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3" #"liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
    image_file = "/nas-ssd2/vaidehi/projects/WikiWeb2M/images-filter/{}"
    load_8bit = False
    model_base = None #"lmsys/vicuna-13b-v1.3"
    device = "cuda"
    conv_mode = None
    temperature = 0.2
    max_new_tokens = 512
    load_4bit = False
    image_aspect_ratio = 'pad'
    split = "train"
    # txt = "Combine the following text with the image content coherently and summarize including content from both the text and the image and compress to length less than input text: {}"
    # txt = "[INST] <image>\nCombine the following text with the image content coherently and summarize including content from both the text and the image and compress to a minimal length of less than 3 sentences such that it captures most salient information from both modalities: {}[/INST]"
    # txt = "[INST] <image>\nCompose a multimodal summary of less than three sentences combining information from both the following text and the image coherently without without attributing the information to text or image in less than three sentences: {} [/INST]"
    txt = "Compose a multimodal summary of less than three sentences combining information from both the following text and the image coherently without without attributing the information to text or image in less than three sentences: {}"

    debug = True
    data_path = "/nas-ssd/vaidehi/LLaVA_v1.6_data/{}_llava_v1.6_7b_zs_summ_all.csv"#"/nas-ssd2/vaidehi/projects/WikiWeb2M/data_v1.6.csv"
    # data_path = "/nas-ssd2/vaidehi/projects/WikiWeb2M/{}_zeroshot_llava_v1.5_13b_all_filt.csv"
    out_path = "/nas-ssd/vaidehi/LLaVA_v1.6_data/llava_v1.6_7b_base_zs_summ_{}_{}.csv"
    img_dir = "/nas-ssd2/vaidehi/projects/WikiWeb2M/images-filter/"
    j = 0
    size = 10000


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    existing_images = os.listdir(args.img_dir)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token 
    data = pd.read_csv(args.data_path.format(args.split))
    data = data[data.img_name.isin(existing_images)]
    print(len(data))
    j, size = args.j, args.size
    data = data[j*size:(j+1)*size]


    


    summaries = []
    # prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        conv_mode = "mistral_instruct"
        args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles    
        
        image = load_image(args.image_file.format(row["img_name"]))
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        inp = args.txt.format(row["txt"])
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None

        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()



        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=None,
                use_cache=True)     

        outputs = tokenizer.decode(output_ids[0]).strip()   
        print(outputs)
        # exit()

        # conv.messages[-1][-1] = outputs
        # # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        # # print(args.txt.format(row["txt"]))
        # input_ids = tokenizer_image_token(args.txt.format(row["txt"]), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # # print(input_ids)
        # # print(tokenizer.batch_decode(input_ids[:, -10:], skip_special_tokens=True)[0])
        # # exit()
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # with torch.inference_mode():
        #     output_ids = model.generate(
        #     input_ids,
        #     images=image_tensor,
        #     do_sample=True,
        #     temperature=0.2,
        #     max_new_tokens=1024,
        #     # use_cache=True,
        #     # stopping_criteria=[stopping_criteria]
        #     )

        # input_token_len = input_ids.shape[1]
        # # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # # if n_diff_input_output > 0:
        # #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # print(outputs)
        # outputs = outputs.strip()
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        # outputs = outputs.strip()
        # print(outputs)
        summaries.append(outputs)
    data["llava_v1.6_7b_base_zs"] = summaries
    data.to_csv(args.out_path.format(args.split, j))


    #     prompt = args.txt.format(row["txt"])


    #     image = Image.open(args.image_file.format(row["img_name"]))
    #     inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    #     try:
    #         output_ids = model.generate(
    #             **inputs,
    #             do_sample=True,
    #             temperature=args.temperature,
    #             max_new_tokens=args.max_new_tokens,
    #             use_cache=True)
    #             # print(inputs.input_ids.shape)
    #         outputs = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    #         print(outputs)
    #         exit()

    #         summaries += [outputs]
    #     except:
    #         summaries += [""]

    # data["llava_v1.6_7b_zs"] = summaries
    # data.to_csv(args.out_path.format(j))


    # qs = args.query
    # if model.config.mm_use_im_start_end:
    #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    # else:
    #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    # else:
    #     args.conv_mode = conv_mode

    # conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()

    # image = load_image(args.image_file)
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=image_tensor,
    #         do_sample=True,
    #         temperature=0.2,
    #         max_new_tokens=1024,
    #         use_cache=True,
    #         stopping_criteria=[stopping_criteria])

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    # print(outputs)
    # return outputs


    # #image_tensor = image_tensor.expand(3, -1, -1, -1)
    # #input_ids = input_ids.expand(3, -1)

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=image_tensor,
    #         do_sample=True,
    #         temperature=0.2,
    #         max_new_tokens=1024,
    #         use_cache=True,
    #         stopping_criteria=[stopping_criteria])

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    # print(outputs)
    # return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--j", type=int, default=0)
    args.j = parser.parse_args().j
    
    eval_model(args)

