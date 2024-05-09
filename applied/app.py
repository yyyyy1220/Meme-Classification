import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import gradio as gr

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

path_or_name = "/home/LLaVA/output/merged_model/meme-llava-v1.5-13b-task-lora"

def text_classification(text):
  # TODO: add trained text classification model here 
  # to predict the sentiment of the text, 
  # save result to result variable 
  
  result = "positive" if "bad" not in text else "negative"
  return result
  
def llava_inference(image_file, model_path = path_or_name):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": "What metaphor is this meme trying to convey?",
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    output = eval_model(args)
    return output

def predict(image_file):
    output = llava_inference(image_file)
    result = text_classification(output)
    return output, result

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Enhanced Meme Classification with VLM")
        
        image_file = gr.Image(label="Please upload the meme")
        infer_btn = gr.Button("Classify Meme")
        metaphor = gr.Textbox(label="Inferred Metaphor", interactive=True)
        result = gr.Textbox(label="Classification Result", interactive=True)

        infer_btn.click(
            predict,
            inputs=[image_file],
            outputs=[metaphor, result]
        )

        demo.launch(share=True)


if __name__ == "__main__":
    main()
