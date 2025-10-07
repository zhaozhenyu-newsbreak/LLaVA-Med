
from PIL import Image
import os
import json
import torch
from torch.utils.data import DataLoader
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer

import logging


class LlavaInference:
    def __init__(self, model_path='/home/fyj/projects/llava-med-v1.5-mistral-7b', 
                 model_name='llava-med-v1.5-mistral-7b', **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_name=model_name,
            model_base=None,
            device=self.device
        )
        self.image_token = DEFAULT_IMAGE_TOKEN
        
    def inference_one(self, prompt, image_paths=None, max_new_tokens=2024):
        num_image_tokens = 0
        max_context_length = 2048
        images = None
        image_args = {}
        if image_paths is not None:
            if self.image_token not in prompt:
                prompt = f"{prompt}{self.image_token * len(image_paths)}\n"
            if len(image_paths) != prompt.count(self.image_token):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
            prompt = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt}
                ], tokenize=False)
            
            print(f"prompt:{prompt}")
            images = [Image.open(image_path) for image_path in image_paths]
            images = process_images(images, self.image_processor, self.model.config)
            if type(images) is list:
                images = [image.to(self.model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(self.model.device, dtype=torch.float16)
            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            num_image_tokens = prompt.count(replace_token) * self.model.get_vision_tower().num_patches
            image_args = {"images": images}
            # image_args = {}
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = ['</s>']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
        if max_new_tokens < 1:
            return "Exceeds max token length. Please start a new conversation, thanks."
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            generate_ids = self.model.generate(inputs=input_ids,
                                               max_new_tokens=max_new_tokens,
                                            #    stopping_criteria=[stopping_criteria],
                                               attention_mask=attention_mask,
                                               **image_args)
            text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=False)
            return text
    
if __name__ == '__main__':
    inference = LlavaInference()
    print(inference.inference_one(prompt='Is there a breast tumor in this image? Please answer yes or no first, and then provide an explanation.', 
                                #   image_paths=None,
                                  image_paths=['/home/fyj/projects/BUS/bus/task-data/zs/processed/29-05-20201123-001/D/89-img.png'],
                                #   image_paths=['/home/fyj/projects/LLaVA-Med/test.png']
                                  ))