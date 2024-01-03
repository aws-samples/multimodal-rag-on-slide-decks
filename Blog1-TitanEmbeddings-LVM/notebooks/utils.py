import os
import boto3
from sagemaker.s3 import S3Uploader
import globals

s3 = boto3.client('s3')

def create_s3_structure():
    s3.put_object(Bucket=BUCKET_NAME, Key=(BUCKET_PREFIX+'/'))
    s3.put_object(Bucket=BUCKET_NAME, Key=(BUCKET_EMB_PREFIX+'/'))
    s3.put_object(Bucket=BUCKET_NAME, Key=(BUCKET_IMG_PREFIX+'/'))

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