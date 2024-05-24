"""
Global variables used throughout the code.
"""
import os
import boto3
import sagemaker

BUCKET_PREFIX: str = "multimodal"
BUCKET_EMB_PREFIX: str = f"{BUCKET_PREFIX}/osi-embeddings-json"
BUCKET_IMG_PREFIX: str = f"{BUCKET_PREFIX}/img"
BUCKET_PDF_IMG_PREFIX: str = f"{BUCKET_PREFIX}/pdf_img"
BUCKET_PDF_TEXT_PREFIX: str = f"{BUCKET_PREFIX}/pdf_text"

S3_MODEL_CODE_PREFIX: str = "code"
S3_MODEL_PREFIX: str = "model"

# Amazon Titan Text model
AWS_REGION: str = boto3.Session().region_name
TITAN_URL: str = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"
TITAN_MODEL_ID: str = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
ACCEPT_ENCODING: str = "application/json"
CONTENT_ENCODING: str = "application/json"

# Amazon OpenSearch Service Serverless
OS_SERVICE: str = "aoss"

# local files and folder structure
IMAGE_DIR: str = "img"
PDF_IMAGE_DIR: str = "pdf_img"
PDF_TEXT_DIR: str = f"{BUCKET_PREFIX}/pdf_txt"
IMAGE_FILE_EXTN: str = ".jpg"
TEXT_FILE_EXTN: str = ".txt"
B64_ENCODED_IMAGES_DIR: str = os.path.join(IMAGE_DIR, "b64_images")

# json files
JSON_TEXT_DIR: str = "pdf_text_json_dir"

# CFN_STACK_NAME: str = "multimodal-blog2-stack"

# prompt to generate questions based on the text/image content provided
QUESTION_GEN_PROMPT: str = """
Human: Based on the text description provided in <text_desc></text_desc> tags, generate a list of five to 10 questions. Only refer to the 
context in the <text_desc> tags, and do not provide questions that are not related to the context provided. 

Your response should be in a JSON format containing two elements: "question" and "answer". The question should be directly related to the 
context provided in the <text_desc> tags and the answer should be the answer to that question from the <text_desc> context. Do not make up an answer.

If you do not know the answer to the question just say that you don't know the answer. Don't try to make up an answer or a question. Refer to the context below:

<text_desc>
{context}
</text_desc>

Assistant: Sure, here are a list of Questions and Answers generated from the context in JSON format:
"""