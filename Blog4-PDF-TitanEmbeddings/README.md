# Talk to your PDF files (Enhanced Multimodal RAG) using foundation models (FMs) hosted on Amazon Bedrock and `Hybrid Search`

This example shows how to use an enhanced RAG workflow to generate responses to user questions from multiple`PDF files`. In this repository, we will generate accurate and specific responses to user questions on texts and images provided in a large corpus of PDF files. This solution is divided into 4 main notebooks, each notebook representing an independent step. Developers/Users can run this repository notebook by notebook, or through a single command via the `command line interface`. This solution contains information about ingesting text and image data from multiple PDF files into `OpenSearch Serverless Service`, Utilizes foundation models on bedrock to store text and image descriptions, convert them into embeddings. This solution also proposes another section of evaluating your own dataset of questions on the PDF files and human generated responses using evaluation frameworks like `ROUGE`, `Cosine Similarity` scores and subjective `LLM as a judge` evaluation metrics. 

Additionally, this repo also shows how to use [LiteLLM](https://github.com/BerriAI/litellm) for interfacing with Bedrock and [Ray](https://github.com/ray-project/ray) for running Bedrock inference concurrently in an asynchronous manner.

## Workflow

The following steps describe how this solution works.

### Data Preparation - Ingest and store PDFs using text and image files

1. In this directory, we provide with an option to use a publicly available [AWS whitepaper]('https://docs.aws.amazon.com/pdfs/whitepapers/latest/ml-best-practices-healthcare-life-sciences/ml-best-practices-healthcare-life-sciences.pdf#ml-best-practices-healthcare-life-sciences') as a PDF file on `machine learning best practices for healthcare life sciences`. If you want to bring your own custom PDF files, or URLs, or a combination of both, then mention them in the ['config.yaml']('config.yaml') file in the `content_info` section under `pdf_local_files`.

1. The [`1_data_prep_pdf_files.ipynb`](Blog4-PDF-TitanEmbeddings/notebooks/1_data_prep_pdf_files.ipynb) notebook handles data preparation for `PDF files`. It utilizes the PDF files available in the `pdf_data` directory, extracts text from each page of the PDF file using the `PyPDF2` library and storing each in a `.txt` file. It converts each page in the PDF file into an image and crops it in 4 parts: `2 horizontal` and `2 vertical` halves and stores it as `.jpg` files based on how many parts a user wants to split the image into and stores the extracted texts and images in an S3 bucket for further analytics and RAG workflow purposes.

The user has the flexibility to choose from the following options to crop the image (or the `pdf page` as an image):

  1. Crop the `pdf image` vertically into two halves: a left half and a right half
  1. Crop the `pdf image` horizontally into two halves: an upper half and a lower half
  1. Crop the `pdf image` both horizontally and vertically in 4 parts
  1. Not crop the `pdf image` at all and save the PDF page as a single image.
  1. Control the image resolution.

  All of these configurations can be set via the `config.yaml` file in the `page_split_imgs` section. For the purpose of this repository, best results for PDF files were found with a whole image stored without cropping it, with the image resolution or `image_scale` set to a high score (in this case, 3).

*This directory has been tested with splitting the images in all possible permutations. For different use cases, structure of PDF files, and content within those files, the best possible way to fix it might change. This directory was tested with PDF files that were all of the similar structure, heavy in text, bars, charts and graphs*

After extracting all of the texts and images from PDF files in separate text and image files, the files are sent to `Amazon S3` that provides a secure location for an enterprise to store these images and a multimodal model can read the texts/images directly from the S3 bucket.

### Data Ingestion - Storing the `Image` and `Text` files in separate indexes as embeddings

1. This portion of the workflow (notebook: [Blog4-PDF-TitanEmbeddings/notebooks/2_data_ingestion.ipynb](Blog4-PDF-TitanEmbeddings/notebooks/2_data_ingestion.ipynb)) downloads the images and text files corresponding to the `pdf file` that we uploaded into Amazon S3 in the [1_data_prep_pdf_files.ipynb](Blog4-PDF-TitanEmbeddings/notebooks/1_data_prep_pdf_files.ipynb) notebook, get text description from `images` and extract text from `text files`, convert both into embeddings and then ingest these embeddings into a vector database i.e. [Amazon OpenSearch Service Serverless](https://aws.amazon.com/opensearch-service/features/serverless/) in two separate indexes. One `text only` index for text based content in the PDF file and one `image only` index for images stored from the file.

1. We use the [Anthropic’s Claude 3 Sonnet foundation model](https://aws.amazon.com/about-aws/whats-new/2024/03/anthropics-claude-3-sonnet-model-amazon-bedrock/) available on Bedrock to convert image to text descriptions.

1. We use the text extracted from each pdf page as is and convert them into embeddings using [Amazon Titan Text Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) and stored in a `text` index. 

1. Each image file is first described using `Claude Sonnet` and then the embeddings of the text description of that image is stored in an `image index`.

1. We use an `entities` field in the `index body metadata` to store entities from both images and texts in their respective `image and text indexes`. The entities from images are extracted using `Claude Sonnet` and entities from texts extracted files using `nltk`. The purpose of extracting these entities is to later use them as a `knn search pre-filter` and use hybrid search based on extracted metadata to improve the quality of the search.

1. The embeddings are then ingested into OpenSearch Service Serverless using the [Amazon OpenSearch Ingestion](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/ingestion.html) pipeline. We ingest the embeddings into an OpenSearch Serverless index via the OpenSearch Ingestion API separately for text and images.

1. Both the image and text content that is posted to the `OSI endpoint` is also stored locally as `json files` containing the text extracted from text files, image description of images, entities for both content types, file names, page numbers and file types for further evaluation and analytical purposes.

An example of a prompt that is used to get entities from an Image using `Claude Sonnet`:

```
entity_extraction_prompt: str = """
Please provide a detailed description of the entities present in the image. Entities, are specific pieces of information or objects within a text that carry particular significance. These can be real-world entities like names of people, places, organizations, or dates. Refer to the types of entities: Named entities: These include names of people, organizations, locations, and dates. You can have specific identifiers within this, such as person names or person occupations.

Custom entities: These are entities specific to a particular application or domain, such as product names, medical terms, or technical jargon.

Temporal entities: These are entities related to time, such as dates, times, and durations.

Product entities: Names of products might be grouped together into product entities.

Data entities: Names of the data and metrics present. This includes names of metrics in charts, graphs and tables, and throughout the image.

Now based on the image, create a list of these entities. Your response should be accurate. Do not make up an answer.
"""
```

*Entities from text files are extracted using [`NLTK`](https://www.nltk.org/)

### Enhanced Multimodal Retrieval Augumented Generation (RAG) Inference

1. This notebook ['3_rag_inference.ipynb'](Blog4-PDF-TitanEmbeddings/notebooks/3_rag_inference.ipynb) performs several steps in a pipeline structure to get accurate results based on user provided questions. This step takes in a user question, and extracts entities from that question. It uses the entities from the question, performs `prefiltering` and extracts the top `k` hits from the index (for both the text and the image index) based on the `entities matching`. This step helps retrieve the most accurate results and relevant docs based on entities that are matches from the question to the `metadata entities` that were stored during the `data ingestion step` for both texts and images.

1. Uses an `LLM in the loop` to go over each `k` hit after vector retrieval, checks for if the answer to the question is given in that hit and if not, moves to the next `hit` until the answer is found. If the answer is not found in any, it returns `I don't know`.

1. Uses an `eval dataset` that a user provides with a question bank (in the [`eval_data directory`](eval_data/) as a `csv`/`xls`/`xlsx` file) containing questions and target responses to those questions. It iterates through each question and queries the `text` and the `image indexes` to look for answers to the questions in the evaluation dataset.

1. During retrieval, answers are searched from both, the `text` and the `image` index to provide a combined answer.

1. The retrieved results are stored in the [`metrics`](metrics/) directory for further evaluations.

### Multimodal RAG Evaluation - `ROUGE`, `Cosine Similarity` & `LLM as a judge`

1. This notebook ['4_rag_evaluation.ipynb'](Blog4-PDF-TitanEmbeddings/notebooks/4_rag_evaluation.ipynb) uses the CSV file generated in the previous notebook to run evaluations on each response from the `image`, `text`, and `combined` indexes.

1. It records the `ROUGE` and `Cosine Similarity` scores. For subjective evaluation, this notebook uses an `LLM as a judge`(in this case, ClaudeV3 Sonnet) in the loop to check for the best match answer given the `target response` and the `questions` provided by the user.

1. Records the results for all kinds of responses from `text only index`, `image only index`, and `combined` (from both the text as well as the image index) from `OpenSearch`

This notebook then gives an overview of an `LLM as a judge pick rate` that shows which index response was picked as the best answer to the user question and why.

## Contents -- WIP

The example consists of 4 Jupyter notebooks and a couple of Python scripts:

WIP
- [`config.yml`](./config.yml) - contains configuration parameters such as directory path, model information etc. for this solution. ***The pricing information in the [`config.yml`](./config.yml) is subject to change, please always confirm Bedrock pricing from the [`Amazon Bedrock Pricing`](https://aws.amazon.com/bedrock/pricing/) page***.

### Dataset

The dataset used in this repo is a synthetically dataset generated through the Bedrock chat playground using the Anthropic Claude 3 Sonnet model with the following prompt. It has been manually edits to get into the exact VTT format.

```{.raw}

Human: create a meeting conversation transacript in the VTT format described here https://en.wikipedia.org/wiki/WebVTT between a group of 5 particle physicists. Generate a 1000 line conversation. Call the physicists Dr. A, Dr. B, Dr. C and so on.

Assistant:
```

## Setup

It is best to create a separate conda environment to run this solution using the commands provided below:

```
conda create --name model_eval_bedrock_py311 -y python=3.11 ipykernel
conda activate model_eval_bedrock_py311
pip install -r requirements.txt -U
```

Use the `model_eval_bedrock_py311` conda environment for all notebooks in this folder.

## Running

Run the following command which will run all the notebooks included in this repo in the correct order.

```{.bash}
rm -rf data/metrics data/title_completions
python main.py
```

You could also run the notebooks manually in the order listed below:

1. [`0_chapterize_data.ipynb`](./0_chapterize_data.ipynb) OR  [`0_already_chapterized_data.ipynb`](./0_already_chapterized_data.ipynb)
1. [`1_generate_chapter_titles.ipynb`](./1_generate_chapter_titles.ipynb)
1. [`2_summarize_metrics.ipynb`](./2_summarize_metrics.ipynb)

### Bring your own chapterization logic

The current solution uses a naive approach to chapterization by treating every Nth (=25, configurable) statement as a chapter boundary. You can easily replace this logic by adding your custom logic in the [`chapterize.py`](./chapterize.py) script.


## Tip

Note that you can use OpenSearch Dashboards to interact with the
OpenSearch API to run quick tests on your index and ingested data.

![](images/ML-os-1.png)

## Cleanup

To avoid incurring future charges, delete the resources. You can do this
by deleting the stack from the CloudFormation console.

![](images/ML-16123-2-cloudformation-delete-stack.png)

## Conclusion

Enterprises generate new content all the time and slide decks are a
common mechanism used to share and disseminate information internally
with the organization and externally with customers or at conferences.
Over time, rich information can remain buried and hidden in non-text
modalities like graphs and tables in these slide decks.

You can use this solution and the power of multimodal FMs such as Titan
Text Embeddings and Claude 3 Sonnet models to discover new information
or uncover new perspectives on content in slide decks. You are welcome
to try different Claude models available on Bedrock by updating the
`CLAUDE_MODEL_ID` in the `globals.py` file.

This is part 2 of a 3-part series. We used Amazon Titan Multimodal
Embeddings and LLaVA models in part 1. Look out for part 3 where we will
compare the approaches from part 1 and part 2.

Portions of this code are released under the Apache 2.0 License as
referenced here: https://aws.amazon.com/apache-2-0/

------------------------------------------------------------------------

## Author bio

<img style="float: left; margin: 0 10px 0 0;" src="images/ML-16123-Amit.jpg">
<b>Amit Arora</b> is an AI and ML Specialist Architect at Amazon Web
Services, helping enterprise customers use cloud-based machine learning
services to rapidly scale their innovations. He is also an adjunct
lecturer in the MS data science and analytics program at Georgetown
University in Washington D.C.

<br><br>

<img style="float: left; margin: 0 10px 0 0;" src="images/ML-16123-Manju.jpg">
<b>Manju Prasad</b> is a Senior Solutions Architect at Amazon Web
Services. She focuses on providing technical guidance in a variety of
technical domains, including AI/ML. Prior to joining AWS, she designed
and built solutions for companies in the financial services sector and
also for a startup. She has worked in all layers of the software stack,
ranging from webdev to databases and has experience in all levels of the
software development lifecycle. She is passionate about sharing
knowledge and fostering interest in emerging talent.

<br><br>

<img style="float: left; margin: 0 10px 0 0;" src="images/ML-16123-Archana.jpg">
<b>Archana Inapudi</b> is a Senior Solutions Architect at AWS,
supporting a strategic customer. She has over a decade of cross-industry
expertise leading strategic technical initiatives. Archana is an
aspiring member of the AIML technical field community at AWS. Prior to
joining AWS, Archana led a migration from traditional siloed data
sources to Hadoop at a health care company. She is passionate about
using technology to accelerate growth, provide value to customers, and
achieve business outcomes.

<br><br>

<img style="float: left; margin: 0 10px 0 0;" src="images/ML-16123-Antara.jpg">
<b>Antara Raisa</b> is an AI and ML Solutions Architect at Amazon Web
Services supporting Strategic Customers based out of Dallas, Texas. She
also has previous experience working with large enterprise partners at
AWS, where she worked as a Partner Success Solutions Architect for
digital native customers.
