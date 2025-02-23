import sys
print(sys.path)
#sys.path.insert(0, "/nas-ssd2/vaidehi/projects/LLaVA/cache/")
import os
# print(os.environ['PYTHONPATH'])
os.environ['TRANSFORMERS_CACHE'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
import argparse
import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# exit()
# torch.cuda.set_device(0) 
import llava
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import requests
from PIL import Image, ImageFile
from io import BytesIO
from transformers import TextStreamer
from transformers import set_seed
ImageFile.LOAD_TRUNCATED_IMAGES = True
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
set_seed(7)



def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    j = args.j
    size = args.size
    disable_torch_init()
    existing_images = os.listdir(args.img_dir)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if args.save:
    	model.save_pretrained(args.save_path)
    	tokenizer.save_pretrained(args.save_path)

    #setting for generation
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    data = pd.read_csv(args.data_path)
    data = data[data.img_name.isin(existing_images)]
    print(len(data))
    data = data[j*size:(j+1)*size]
    
    summaries = []
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image = load_image(args.image_file.format(row["img_name"]))
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    

    
        try:
            # inp = input(f"{roles[0]}: ")
            inp = args.txt.format(row["txt"])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print("prompt")
        # print(prompt)


        # prompt =  "Combine the following text with the image content and summarize including content from both image and text: {}".format(args.txt)
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        if input_ids.shape[1]>512:
            input_ids = input_ids[:,-512:]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            # print(input_ids.device)
            # print(image_tensor.device)
            # print(model.device)
            print(input_ids.shape)
            # print(image_tensor.shape)
            # image_tensor = image_tensor.expand(3, -1, -1, -1)
            # input_ids = input_ids.expand(3, -1)


            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        summaries += [outputs]

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        # return outputs

    data["llava_v1.5_7b_tifa_ft_fp_filter_11_round2"] = summaries
    data.to_csv("/nas-ssd2/vaidehi/projects/WikiWeb2M/test_llava_v1.5_7b_tifa_ft_fp_filter_12_round2_{}.csv".format(j))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--txt", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
