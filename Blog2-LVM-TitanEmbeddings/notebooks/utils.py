"""
Utility functions for S3 and CloudFormation used by the rest of the code
"""
import os
import boto3
import logging
import globals as g
from typing import List
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