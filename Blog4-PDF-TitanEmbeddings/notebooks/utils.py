"""
Utility functions for S3 and CloudFormation used by the rest of the code
"""
import os
import yaml
import json
import boto3
import logging
import botocore
import numpy as np
import globals as g
from pathlib import Path
from typing import List, Dict
from litellm import completion
from sagemaker.s3 import S3Uploader

logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

def _merge_dicts(src, dest):
    for key, value in src.items():
        # check for if the key in the user config file is in the parent config file
        if isinstance(value, dict) and key in dest:
            # if there, then call the function again and if not, add the value to the user config
            # file and merge both to return the full config file
            _merge_dicts(value, dest[key])
        else:
            dest[key] = value
                
def load_and_merge_configs(config_path: str, full_config_path: str):
    """
    Load the user config and merge it with the full config.
    
    :param config_path: Path to the user config file.
    :param full_config_path: Path to the full config file.
    :return: Merged configuration dictionary.
    """
    with open(full_config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)

    _merge_dicts(user_config, full_config)
    return full_config

# initialize the full config file
config = load_and_merge_configs(g.CONFIG_SUBSET_FILE, g.FULL_CONFIG_FILE)

def upload_to_s3(local_file_path:str, bucket_name: str, bucket_prefix:str) -> None:
    global s3
    try:
        with open(local_file_path, 'rb') as file:
            s3_key = os.path.join(bucket_prefix, os.path.basename(local_file_path))
            s3.upload_fileobj(file, bucket_name, s3_key)
            logger.info(f"File {local_file_path} uploaded to {bucket_name}/{s3_key}.")
    except Exception as e:
        logger.error(f"error uploading file to S3: {e}")


def download_image_files_from_s3(bucket:str, bucket_image_prefix:str, local_image_dir:str, image_file_extn:str) -> List:
    images = s3.list_objects_v2(Bucket=bucket, Prefix=bucket_image_prefix)['Contents']
    local_file_paths: List = []
    for img in images:
        if img['Key'].endswith(image_file_extn):
            file_path = os.path.join(local_image_dir, os.path.basename(img['Key']))            
            s3.download_file(bucket, img['Key'], file_path)
            logger.info(f"downloaded {bucket}/{img['Key']} to {file_path}")
            local_file_paths.append(file_path)
    return local_file_paths 


def get_cfn_outputs(stackname: str) -> List:  
    cfn = boto3.client('cloudformation')
    outputs = {}
    stacks = cfn.describe_stacks(StackName=stackname)['Stacks']
    if stacks == []:
        return None
    for output in stacks[0]['Outputs']:
        outputs[output['OutputKey']] = output['OutputValue']
    return outputs


def get_bucket_name(stackname: str) -> str:
    outputs = get_cfn_outputs(stackname)
    bucketname = outputs['BucketName']
    return bucketname


def get_text_embedding(bedrock: botocore.client, prompt_data: str) -> np.ndarray:
    body = json.dumps({
        "inputText": prompt_data,
    })
    try:
        response = bedrock.invoke_model(
            body=body, modelId=g.TITAN_MODEL_ID, accept=g.ACCEPT_ENCODING, contentType=g.CONTENT_ENCODING
        )
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')
    except Exception as e:
        logger.error(f"exception={e}")
        embedding = None

    return embedding


def get_llm_response(question: str, 
                     summary: str, 
                     modelId: str = g.CLAUDE_MODEL_ID) -> Dict:
    """
    This function takes in the prompt that checks whether the text file has a response to the question and if not, 
    returns "not found" to move to the next hit.
    """
    get_final_llm_response_prompt_fpath: str = os.path.join(config['dir_info']['prompt_dir'],
                                                                     config['dir_info']['final_combined_llm_response_prompt'])
    final_llm_prompt: str = Path(get_final_llm_response_prompt_fpath).read_text()
    prompt = final_llm_prompt.format(question=question, summary=summary)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    temperature = config['inference_parameters'].get('temperature', 0.1)
    max_tokens = config['inference_parameters'].get('max_tokens', 500)
    # suppress the litellm logger responses
    lite_llm_logger = logging.getLogger('LiteLLM')
    lite_llm_logger.setLevel(logging.CRITICAL)
    ret = {
        "exception": None,
        "prompt": prompt,
        "completion": None,
        "completion_token_count": None,
        "prompt_token_count": None,
        "model_id": modelId,
        "time_taken_in_seconds": None,
        "input_token_cost": None,
        "output_token_cost": None,
    }
    try:
        response = completion(
            model=modelId,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Suppress logging output
        logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
        # iterate through the entire model response
        for idx, choice in enumerate(response.choices):
            # extract the message and the message's content from litellm
            if choice.message and choice.message.content:
                # extract the response from the dict
                ret["completion"] = choice.message.content.strip()
        # Extract number of input and completion prompt tokens (this is the same structure for embeddings and text generation models on Amazon Bedrock)
        ret['prompt_token_count'] = response.usage.prompt_tokens
        ret['completion_token_count'] = response.usage.completion_tokens
        # Extract latency in seconds
        latency_ms = response._response_ms
        ret['time_taken_in_seconds']  = latency_ms / 1000
    except Exception as e:
        logger.error(f"exception={e}")
        ret["exception"] = e
    return ret


def get_question_entities(bedrock: botocore.client, 
                   question:str, 
                   modelId: str = g.CLAUDE_MODEL_ID) -> str:
    get_entities_from_user_question_prompt_fpath: str = os.path.join(config['dir_info']['prompt_dir'],
                                                                     config['dir_info']['extract_entities_from_user_question'])
    question_entities_extraction_prompt: str = Path(get_entities_from_user_question_prompt_fpath).read_text()
    prompt = question_entities_extraction_prompt.format(question=question)

    body = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    })

    try:
        response = bedrock.invoke_model(
        modelId=modelId,
        body=body)

        response_body = json.loads(response['body'].read().decode("utf-8"))
        combined_llm_response = response_body['content'][0]['text'].replace('"', "'")

    except Exception as e:
        logger.error(f"exception while getting question entities: exception={e}")
        combined_llm_response = None

    return combined_llm_response