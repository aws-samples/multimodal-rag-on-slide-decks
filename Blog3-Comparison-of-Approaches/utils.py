"""
Utility functions for S3 and CloudFormation used by the rest of the code
"""
import os
import json
import boto3
import base64
import logging
import botocore
import numpy as np
import globals as g
import requests as req
from pathlib import Path
from typing import List, Dict
from opensearchpy import OpenSearch
from sagemaker.s3 import S3Uploader

logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

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
            
def download_image_from_url(url, image_dir):
    logger.info(f"downloading image at {url}")
    local_file: str = os.path.join(image_dir, os.path.basename(url))
    r = req.get(url, allow_redirects=True)
    if r.status_code == 200:
        logger.info(f"{url} downloaded successfully")
        with open(local_file, "wb") as f:
            f.write(r.content)
        logger.info(f"{url} written to {local_file}")
        return local_file
    return ""

def encode_image_to_base64(image_file_path: str) -> str:
    with open(image_file_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode('utf8')
        b64_image_path = os.path.join(g.B64_ENCODED_IMAGES_DIR, f"{Path(image_file_path).stem}.b64")
        with open(b64_image_path, "wb") as b64_image_file:
            b64_image_file.write(bytes(b64_image, 'utf-8'))
    return b64_image_path

def find_similar_data(os_client: OpenSearch, text_embeddings: np.ndarray, size: int, index_name: str, deck_name: str, deck_url: str) -> Dict:
    query = {
        "size": size,
        "query": {
            "knn": {
                "vector_embedding": {
                    "vector": text_embeddings,
                    "k": size
                }
            }
        }
    }

    try:
        image_based_search_response = os_client.search(body=query, index=index_name)
        logger.info("received response from OpenSearch")
    except Exception as e:
        logger.error(f"error occured while querying OpenSearch index={index_name}, exception={e}")
        image_based_search_response = None
    return image_based_search_response

def get_img_desc(bedrock: botocore.client, image_file_path: str, prompt: str):
    # read the file, MAX image size supported is 2048 * 2048 pixels
    with open(image_file_path, "rb") as image_file:
        input_image_b64 = image_file.read().decode('utf-8')
  
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": input_image_b64
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
    )
    
    response = bedrock.invoke_model(
        modelId=g.CLAUDE_MODEL_ID,
        body=body
    )

    resp_body = json.loads(response['body'].read().decode("utf-8"))
    resp_text = resp_body['content'][0]['text'].replace('"', "'")

    return resp_text

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

def get_text_embedding(bedrock: botocore.client, prompt_data: str, modelID: str) -> np.ndarray:
    body = json.dumps({
        "inputText": prompt_data,
    })    
    try:
        response = bedrock.invoke_model(
            body=body, modelId=modelID, accept=g.ACCEPT_ENCODING, contentType=g.CONTENT_ENCODING
        )
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')
    except Exception as e:
        logger.error(f"exception={e}")
        embedding = None

    return embedding
