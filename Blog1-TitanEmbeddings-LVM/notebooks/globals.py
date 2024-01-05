import os
import boto3
from sagemaker.session import Session

# global constants
HF_MODEL_ID = "anymodality/llava-v1.5-7b"
HF_MODEL_NAME = "model_llava-v1.5-7b.tar.gz"
HF_TASK = "question-answering"
TRANSFORMERS_VERSION = "4.28.1"
PYTORCH_VERSION = "2.0.0"
PYTHON_VERSION = "py310"

BUCKET_NAME = Session().default_bucket()
BUCKET_PREFIX = "multimodal"
BUCKET_EMB_PREFIX = "multimodal/osi-embeddings-json"
BUCKET_IMG_PREFIX = "multimodal/img"

AWS_REGION = boto3.Session().region_name
OS_SERVICE = "aoss"
OS_INDEX = "multimodalslidesindex"

IMAGE_DIR = "img"
B64_ENCODED_IMAGES_DIR = os.path.join(IMAGE_DIR, "b64_images")

FMC_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"
FMC_MODEL_ID = "amazon.titan-embed-image-v1"
ACCEPT_ENCODING = "application/json"
CONTENT_ENCODING = "application/json"

S3_MODEL_URI=os.path.join("s3://", BUCKET_NAME, BUCKET_PREFIX, os.path.basename(HF_MODEL_ID), HF_MODEL_NAME)
SLIDE_DECK = "https://d1.awsstatic.com/events/Summits/torsummit2023/CMP301_TrainDeploy_E1_20230607_SPEdited.pdf"
