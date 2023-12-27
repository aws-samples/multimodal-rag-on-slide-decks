import os
import boto3
from sagemaker.s3 import S3Uploader
from sagemaker.session import Session

# global constants
HF_MODEL_ID="llava-v1.5-7b"
HF_MODEL_NAME="model_llava-v1.5-7b.tar.gz"

BUCKET_NAME = Session().default_bucket()
BUCKET_PREFIX:str = "multimodal"
BUCKET_EMB_PREFIX:str = "multimodal/osi-embeddings-json"
BUCKET_IMG_PREFIX:str = "multimodal/img"

IMAGE_DIR: str = "img"
B64_ENCODED_IMAGES_DIR:str = os.path.join(IMAGE_DIR, "b64_images")

FMC_URL: str = "https://bedrock-runtime.us-east-1.amazonaws.com"
FMC_MODEL_ID: str = "amazon.titan-embed-image-v1"
ACCEPT_ENCODING: str = "application/json"
CONTENT_ENCODING: str = "application/json"
ROLE_ARN="arn:aws:iam::205088436647:role/SMNotebookForMultiModalRAG"

S3_MODEL_URI=os.path.join("s3://", BUCKET_NAME, HF_MODEL_ID, HF_MODEL_NAME)
SLIDE_IMAGE_URL_TEMPLATE: str = "https://raw.githubusercontent.com/aarora79/multimodal/main/amz_trainium/CMP301_TrainDeploy_E1_20230607_SPEdited_image_{}.jpg"
MAX_SLIDES: int = 1
SLIDE_DECK:str = "https://d1.awsstatic.com/events/Summits/torsummit2023/CMP301_TrainDeploy_E1_20230607_SPEdited.pdf"


s3 = boto3.client('s3')
def upload_to_s3(local_file_path:str, bucket_prefix:str):
    try:
        with open(local_file_path, 'rb') as file:
            s3_key = os.path.join(bucket_prefix, os.path.basename(local_file_path))
            s3.upload_fileobj(file, BUCKET_NAME, s3_key)
            logger.info(f"File {local_file_path} uploaded to {BUCKET_NAME}/{s3_key}.")
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        
def download_from_s3():
    images = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=BUCKET_IMG_PREFIX)['Contents']
    for img in images:
        if img['Key'].endswith('.jpg'):
            file_path = os.path.join(IMAGE_DIR, os.path.basename(img['Key']))
            s3.download_file(BUCKET_NAME, img['Key'], file_path)