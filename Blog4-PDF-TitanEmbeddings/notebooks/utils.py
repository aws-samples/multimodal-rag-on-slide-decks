"""
Utility functions for S3 and CloudFormation used by the rest of the code
"""
import os
import json
import boto3
import logging
import botocore
import numpy as np
import globals as g
from typing import List
from sagemaker.s3 import S3Uploader

logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

question_entities_extraction_prompt: str = """

Human: Your role is to extract entities from a question and what is specifically needed from the question. Entities, are specific pieces of information or objects within a text that carry particular significance. These can be real-world entities like names of people, places, organizations, or dates. Refer to the types of entities: Named entities: These include names of people, organizations, locations, and dates. You can have specific identifiers within this, such as person names or person occupations.

    Custom entities: These are entities specific to a particular application or domain, such as product names, medical terms, or technical jargon.

    Temporal entities: These are entities related to time, such as dates, times, and durations.

    Product entities: Names of products might be grouped together into product entities.

    Location entities: These entities categorize or classify items based on location indicators, such as state codes.

Now, refer to the question below in the <question></question> tags and give the entities within it.

<question>
{question}
</question>

Your response should be concise and only contain the names of the entities, nothing else. Be accurate. Do not make up an answer. Do not give titles or headings for the entities. Just give the entities. Your response should NOT contain any filler words like "custom entities: ", and so on. Just give the name of the entities in your response. Your response should have a commas between the words/entities. only spaces between each entity name. View an example below of what a response should be (in <response></response> tags) versus should not be (in <should_not_be></should_not_be>:

<should_not_be>
Named entities: ratings amazon
</should_not_be>

<response>
ratings amazon
</response>

Assistant: Sure, based on the context, here are the names of entities without any filler words before: """

llm_prompt: str = """

Human: Use the context in the <summary></summary> tags to provide a answer to the question to the best of your abilities. If you cannot answer the question from the context then say I do not know, do not make up an answer.

<question>
{question}
</question>

<summary>
{summary}
</summary>

Assistant: Here is my answer based on the context provided:"""


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

def get_llm_response(bedrock: botocore.client, 
                     question: str, 
                     summary: str, 
                     modelId: str = g.CLAUDE_MODEL_ID) -> str:
    prompt = llm_prompt.format(question=question, summary=summary)

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
        llm_response = response_body['content'][0]['text'].replace('"', "'")

    except Exception as e:
        logger.error(f"exception while slide_text={summary[:10]}, exception={e}")
        llm_response = None

    return llm_response

def get_question_entities(bedrock: botocore.client, 
                   question:str) -> str:
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
        modelId=g.CLAUDE_MODEL_ID,
        body=body)

        response_body = json.loads(response['body'].read().decode("utf-8"))
        combined_llm_response = response_body['content'][0]['text'].replace('"', "'")

    except Exception as e:
        logger.error(f"exception while slide_text={summary[:10]}, exception={e}")
        combined_llm_response = None

    return combined_llm_response