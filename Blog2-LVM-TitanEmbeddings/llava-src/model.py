import os
import boto3
import json
import torch
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from djl_python import Input, Output

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# from transformers import TextStreamer

model_dict = None


class ModelConfig:
    def __init__(self):
        image_aspect_ratio = "pad"


def divide_bucket_key_and_filename_from_s3_uri(s3_uri):
    # Parse the S3 URI
    parsed_uri = urlparse(s3_uri)

    # Extract the bucket name
    bucket = parsed_uri.netloc

    # Extract the key (object path)
    key = parsed_uri.path.lstrip("/")

    # Extract the filename
    filename = key.split("/")[-1]
    return bucket, key, filename
    
        
        
def load_model(properties):
    s3 = boto3.client('s3')
    
    disable_torch_init()
    
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    
    model_path = model_location
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None, model_name="llava", load_8bit=True)
    model_cfg = ModelConfig()
    model_dict = {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "model_cfg": model_cfg,
        "s3": s3
    }
    
    return model_dict
    

def handle(inputs: Input):
    global model_dict
    
    conv_mode = "llava_v1"
    temperature = 0.1
    max_new_tokens = 512
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    image_processor = model_dict['image_processor']
    model_cfg = model_dict['model_cfg']
    s3 = model_dict['s3']
    
    data = inputs.get_as_json()
    inp = data["text"][0]
    input_image_s3 = data["input_image_s3"]
    bucket, key, fname = divide_bucket_key_and_filename_from_s3_uri(input_image_s3)
    
    local_dir = Path("/tmp/image-input")
    local_dir.mkdir(exist_ok=True)
    local_file_path = os.path.join(local_dir, fname)
    s3.download_file(Bucket=bucket, Key=key, Filename=local_file_path)
    print(f"Local image path: {local_file_path}")    
    
    raw_image = Image.open(local_file_path).convert('RGB')
    image_tensor = process_images([raw_image], image_processor, model_cfg)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    if raw_image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        raw_image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(prompt)
    
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=None,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
    return Output().add(output)
                