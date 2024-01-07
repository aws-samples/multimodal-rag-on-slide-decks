## Blog1-TitanEmbeddings-LVM
### This folder contains the following files:

* `0_modelprep.py` - In this notebook we prepare The LLaVA-1.5 model to be deploy it as a customized model on an Amazon SageMaker endpoint by creating a `model.tar.gz` containing inference.py and model weights (one-time type activity).
* `1_dataprep.py` - In this notebook we download a publicly available slide deck and convert it into images, one image for each slide. These images are then stored in Amazon S3 from where they can be made available to a Amazon SageMaker Endpoint for inference.
* `2_dataingestion.py` - In this notebook we do the data ingestion where we first takes files from Amazon S3, then use the Amazon Bedrock Titan Multimodal Model to return embeddings as JSON to store in S3, and finally call the Amazon OpenSearch Ingestion Pipeline to store the embeddings in a vector database in OpenSearch. 
* `3_raginference.py` - In this notebook we take a user prompt and convert it into embeddings by the Titan Multimodal Embeddings model. Then we do an OpenSearch vector search using the prompt embeddings. Finally we use the image returned by from the vector search response by passing it to LLaVA for inference and return the response to the user

* `template.yaml` which is the CloudFormation template that creates the following resources:
  * SageMaker Notebook
  * OpenSearch Collection
  * OpenSearch Ingestion (OSI) Pipeline
  * SQS Queue

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

