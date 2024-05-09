from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


fine_tuned_model_path = "/home/bingxuan/bingxuan/LLaVA/checkpoints/llava-v1.5-13b-task-lora"
from llava.model.builder import load_pretrained_model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=fine_tuned_model_path,
    model_base="liuhaotian/llava-v1.5-13b", 
    model_name=get_model_name_from_path(fine_tuned_model_path)
)

prompt = "What metaphor is this meme trying to convey?"
image_file = "/home/bingxuan/bingxuan/data/meme_data/imgs/test/memes_k4xv8c.png"

args = type('Args', (), {
    "model_path": fine_tuned_model_path,
    "model_base": "liuhaotian/llava-v1.5-13b",
    "model_name": get_model_name_from_path(fine_tuned_model_path),
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