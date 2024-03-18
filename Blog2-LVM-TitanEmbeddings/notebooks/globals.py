"""
Global variables used throughout the code.
"""
import os
import boto3
import sagemaker

# S3 bucket strucutre, we use the default sagemaker bucket in the current region
# BUCKET_NAME: str = sagemaker.Session().default_bucket()
BUCKET_PREFIX: str = "multimodal"
BUCKET_EMB_PREFIX: str = f"{BUCKET_PREFIX}/osi-embeddings-json"
BUCKET_IMG_PREFIX: str = f"{BUCKET_PREFIX}/img"
LLAVA_PROMPT_PREFIX: str = 'multimodal/llavaPrompt/'

S3_MODEL_CODE_PREFIX: str = "code"
S3_MODEL_PREFIX: str = "model"

# Amazon Titan multimodal model
AWS_REGION: str = boto3.Session().region_name
FMC_URL: str = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"
FMC_MODEL_ID: str = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
ACCEPT_ENCODING: str = "application/json"
CONTENT_ENCODING: str = "application/json"

# Amazon OpenSearch Service Serverless
OS_SERVICE: str = "aoss"

# local files and folder structure
IMAGE_DIR: str = "img"
IMAGE_FILE_EXTN: str = ".jpg"
B64_ENCODED_IMAGES_DIR: str = os.path.join(IMAGE_DIR, "b64_images")

# this is the slide deck to which we will be talking to. Replace with your slide deck's URL to analyze a different deck
SLIDE_DECK: str = "https://d1.awsstatic.com/events/Summits/torsummit2023/CMP301_TrainDeploy_E1_20230607_SPEdited.pdf"

# AWS CloudFormation stack that created the resources for this blog post including this notebook
# if a different name is used while creating the CloudFormation stack then change this to match the name you used
CFN_STACK_NAME: str = "multimodal-blog2-stack"

