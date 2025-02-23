import argparse
import os
os.environ['TRANSFORMERS_CACHE'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
os.environ['HF_HOME'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
os.environ['TRITON_CACHE_DIR'] = '/nas-ssd2/vaidehi/projects/LLaVA/cache/'
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
