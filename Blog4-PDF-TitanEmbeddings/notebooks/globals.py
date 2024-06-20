"""
Global variables used throughout the code.
"""
import os
import boto3
import sagemaker
from typing import List

BUCKET_PREFIX: str = "multimodal"
BUCKET_EMB_PREFIX: str = f"{BUCKET_PREFIX}/osi-embeddings-json"
BUCKET_IMG_PREFIX: str = f"{BUCKET_PREFIX}/img"
BUCKET_TEXT_PREFIX: str = f"{BUCKET_PREFIX}/text"

# Path to the config files
CONFIG_FOLDER: str = "configs"
CONFIG_SUBSET_FILE: str = "config.yaml"
FULL_CONFIG_FILE = f"{CONFIG_FOLDER}/config_full.yaml"

S3_MODEL_CODE_PREFIX: str = "code"
S3_MODEL_PREFIX: str = "model"

# Amazon Titan Text model
TITAN_MODEL_ID: str = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
ACCEPT_ENCODING: str = "application/json"
CONTENT_ENCODING: str = "application/json"

# Amazon OpenSearch Service Serverless
OS_SERVICE: str = "aoss"

# local files and folder structure
IMAGE_DIR: str = "img"
LOCAL_IMAGE_DIR: str = f"{BUCKET_PREFIX}/local_imgs"
LOCAL_TEXT_DIR: str = f"{BUCKET_PREFIX}/local_txts"
TEXT_DIR: str = "text"
IMAGE_FILE_EXTN: str = ".jpg"
TEXT_FILE_EXTN: str = ".txt"
B64_ENCODED_IMAGES_DIR: str = os.path.join(IMAGE_DIR, "b64_images")

# Qualitative metrics on the following list
QUALITATIVE_METRICS_LIST: List[str] = ['combined_response', 'text_response', 'img_response']

# other global values
IMAGE_FORMAT: str = "JPEG"
BEDROCK_EP_URL: str = "https://bedrock-runtime.{region}.amazonaws.com"