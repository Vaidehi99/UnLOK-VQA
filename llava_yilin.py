from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
from PIL import Image
import requests
from io import BytesIO
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

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


model_path = "liuhaotian/llava-v1.5-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

image_file = ""
image = load_image("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/download.webp")
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device)

# inputs = tokenizer(["Who is this person?"], return_tensors="pt").to(device)
inputs = tokenizer_image_token("<image> Who is this person?", tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
# inputs = inputs.view(1,-1)
print(inputs)
outputs = model.generate(
   inputs, 
   images=image_tensor,
   max_new_tokens=50, 
   return_dict_in_generate=True, 
   output_scores=True
)

print(tokenizer.batch_decode(outputs[0]))
print(tokenizer("Brazil"))

# import pdb; pdb.set_trace()

#print(outputs)

# print("image_tensor.shape: ", image_tensor.shape)

# print("inputs.input_ids.shape: ", inputs.input_ids.shape)

# print("model(inputs.input_ids, images = image_tensor).logits.shape: ", model(inputs.input_ids, images = image_tensor).logits.shape)
