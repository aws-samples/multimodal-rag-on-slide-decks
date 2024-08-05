"""
Global variables used throughout the code.
"""
import os
import boto3
import sagemaker

BUCKET_PREFIX: str = "multimodal"
BUCKET_EMB_PREFIX: str = f"{BUCKET_PREFIX}/osi-embeddings-json"
BUCKET_IMG_PREFIX: str = f"{BUCKET_PREFIX}/img"

S3_MODEL_CODE_PREFIX: str = "code"
S3_MODEL_PREFIX: str = "model"

# Amazon Titan Text model
AWS_REGION: str = boto3.Session().region_name
TITAN_URL: str = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"
TITAN_MODEL_ID: str = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
ACCEPT_ENCODING: str = "application/json"
CONTENT_ENCODING: str = "application/json"

# Amazon Titan multimodal model
FMC_MODEL_ID: str = "amazon.titan-embed-image-v1"

# Amazon OpenSearch Service Serverless
OS_SERVICE: str = "aoss"

# local files and folder structure
IMAGE_DIR: str = "img"
IMAGE_FILE_EXTN: str = ".jpg"
B64_ENCODED_IMAGES_DIR: str = os.path.join(IMAGE_DIR, "b64_images")
EMBEDDINGS_DIR: str = "embeddings"

# AWS CloudFormation stack that created the resources for this blog post including this notebook
# if a different name is used while creating the CloudFormation stack then change this to match the name you used
CFN_STACK_NAME: str = "multimodal-blog2-stack"

