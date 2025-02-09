from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)


model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What is the street address of this place?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"
image_file = "https://static01.nyt.com/images/2015/04/19/travel/20150419EUROPEANSTREETS-slide-G82E/20150419EUROPEANSTREETS-slide-G82E-master1050.jpg"
image_file = "https://upload.wikimedia.org/wikipedia/commons/3/33/Old_Bond_Street_2_db.jpg"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
