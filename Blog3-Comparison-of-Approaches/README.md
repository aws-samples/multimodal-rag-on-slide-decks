# Talk to your slide deck (Multimodal RAG) using foundation models (FMs) hosted on Amazon Bedrock – Part 3

In the "Talk to your slide deck" blog series, we share two different approaches to interact with data stored in text, images, graphs, and charts. In [part
1](https://aws.amazon.com/blogs/machine-learning/talk-to-your-slide-deck-using-multimodal-foundation-models-hosted-on-amazon-bedrock-and-amazon-sagemaker-part-1/), we presented a solution that used [Amazon Titan Multimodal Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html) model to convert individual slides from a slide deck into embeddings. We stored the embeddings in a vector database and then used the [Large
Language-and-Vision Assistant (LLaVA 1.5-7b)](https://llava-vl.github.io/) model to generate text responses
to user questions based on the most similar slide retrieved from the vector database. This solution is what we refer to as "embed first, infer later". Part 1 uses AWS services including [Amazon Bedrock](https://aws.amazon.com/bedrock/), [Amazon SageMaker](https://aws.amazon.com/sagemaker/), and [Amazon OpenSearch Serverless](https://aws.amazon.com/opensearch-service/features/serverless/).

In [part 2](https://aws.amazon.com/blogs/machine-learning/talk-to-your-slide-deck-using-multimodal-foundation-models-hosted-on-amazon-bedrock-and-amazon-sagemaker-part-2/), we demonstrated a different approach (infer first, embed later). We used [Anthropic Claude 3 Sonnet](https://aws.amazon.com/bedrock/claude/)
model to generate text descriptions for each slide in the slide deck. These descriptions are then converted into text embeddings using [Amazon Titan Text Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) model and stored in a vector database. Then we used the Claude 3 Sonnet
model to generate answers to user questions based on the most relevant text description retrieved from the vector database.

In this post, we evaluate the results from both approaches using a single source of truth. You can test both approches and evaluate the results to find the best fit for your datasets. The code for this series is available in the [GitHub](https://github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main) repo.

## Solution overview

Please refer to Solution overview and design in Parts 1 & 2 of the series. 

## Comparison

[SlideVQA](https://github.com/nttmdlab-nlp/SlideVQA) is a visual question answering dataset. It is a collection of publicly available slide decks, each composed of multiple slides (in jpg format) and a corresponding question. It allows a system to select a set of evidence images and answer the question.

### Results

We executed the solutions for Part 1 and Part 2 on a [sample manifest](qa.jsonl) created from SlideVQA. 

<I>Note: The responses to the questions in the sample manifest are concise in as few words as possible. We updated the prompts in each approach to provide precise responses instead of verbose answers. This helped in comparing the responses to the ground truth. </I>

Below sections will briefly discuss the solutions and dive into the evaluation and pricing for each approach.

#### Approach 1 (embed first, infer later)

Slide decks are converted into pdf images, one per slide, and embedded using the Titan Multimodal Embeddings model, resulting in a vector embedding of 1,024 dimensions. The embeddings are stored in OpenSearch Serverless index which serves as the vector store for our RAG solution. The embeddings are ingested via [Amazon OpenSearch Ingestion Pipeline](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/ingestion.html) (OSI).

Each question is converted into embeddings using the Titan Multimodal Embeddings model and an OpenSearch vector search is performed using these embeddings. We performed a k-nearest neighbor (knn) search to retrieve the most relevant embedding matching the question. The metadata of the response from OpenSearch index contains a path to the image corresponding to the most relevant slide.

A prompt is created by combining the question and the image path and sent to LLaVA 1.5-7b to respond to the question with a precise answer.

##### Evaluation

A precise response for each question in the manifest is recorded and compared to the ground truth provided by SlideVQA. 

<I> Note: We used Claude 3 Sonnet instead of LLaVA 1.5-7b as mentioned in the solution for Part 1. The approach remains the same, embed first and infer later, just the model that compiles the final response is changed for simplicity. </I>

Using approach 1, we received about 50% accurate results to the questions in the manifest.

#### Approach 2 (infer first, embed later)

Slide decks are converted into pdf images, one per slide, and passed to the Claude 3 Sonnet model to generate a text description. The description is sent to the Titan Text Embeddings model to generate vector embeddings with 1,536 dimensions. The embeddings are ingested into OpenSearch Serverless index via OSI pipeline. 

Each question is converted into embeddings using the Titan Text Embeddings model and an OpenSearch vector search is performed using these embeddings. We performed a k-nearest neighbor (knn) search to retrieve the most relevant embedding matching the question. The metadata of the response from OpenSearch index contains the image description corresponding to the most relevant slide.

We create a prompt with the question and the image description and pass it to Claude 3 Sonnet to receive a precise answer.

##### Evaluation

A precise response for each question in the manifest is recorded and compared to the ground truth provided by SlideVQA. 

With approach 2, infer first and embed later, we received 44% accurate results for questions in our sample manifest.

#### Analysis of results

In our testing, both approaches produced 50% or less accurate results to the questions in the manifest. Our random selection of slide decks covered a wide variety of industries including retail, healthcare, academic, technology, personal, travel etc. The embeddings were all ingested into a single index. So a generic question like "What are examples of tools that can be used?" would have no additional context. The closest match can be retrieved from a completely different slide deck, hence not matching the accurate response. We will suggest ways these results can be improved in the conclusion below.

The final prompt to Claude 3 Sonnet in our analysis included instructions to provide a precise answer in as few words as possible to be able to compare with the ground truth. We assume your responses will depend on your prompts to the large language model.

### Pricing

Pricing is dependent on the modality, provider, and model used. Please refer to the detailed public pricing for Amazon Bedrock [here](https://aws.amazon.com/bedrock/pricing/). We use the On-Demand and Batch pricing mode in our analysis. It allows you to use FMs on a pay-as-you-go basis without having to make any time-based term commitments. For text-generation models, you are charged for every input token processed and every output token generated. For embeddings models, you are charged for every input token processed.

The below table shows price per question for each approach. 

<I> Note: We calculated the average number of input and output tokens based on our sample dataset for us-east-1, pricing may vary for your datasets and Amazon region used. </I>

|                               	|                                   	|      **Approach   1**     	|                 	|              	|                           	|                 	|                  	|
|-------------------------------	|-----------------------------------	|:-------------------------:	|:---------------:	|:------------:	|:-------------------------:	|:---------------:	|------------------	|
|                               	|                                   	|      **Input tokens**     	|                 	|              	|     **Output tokens**     	|                 	|                  	|
| **Model**                     	| **Description**                   	| **Price per 1000 tokens** 	| **# of tokens** 	| **Price**    	| **Price per 1000 tokens** 	| **# of tokens** 	| **Price**        	|
| Titan   Multimodal Embeddings 	| Slide/image embedding             	| $0.0001                   	| 1               	| $0.000000    	| $0.0000                   	| 0               	| $0.000000        	|
| Titan   Multimodal Embeddings 	| Question embedding                	| $0.0001                   	| 1               	| $0.000000    	| $0.0000                   	| 0               	| $0.000000        	|
| Claude 3   Sonnet             	| Final response                    	| $0.0030                   	| 700             	| $0.002100    	| $0.0150                   	| 8               	| $0.000120        	|
| Cost per   input/output       	|                                   	|                           	|                 	| $0.002100    	|                           	|                 	| $0.000120        	|
| **Total cost per   question** 	|                                   	|                           	|                 	|              	|                           	|                 	| **$0.002220** 	|
|                               	|                                   	|                           	|                 	|              	|                           	|                 	|                  	|
|                               	|                                   	|       **Approach 2**      	|                 	|              	|                           	|                 	|                  	|
|                               	|                                   	|      **Input tokens**     	|                 	|              	|     **Output tokens**     	|                 	|                  	|
| **Model**                     	| **Description**                   	| **Price per 1000 tokens** 	| **# of tokens** 	| **Price**    	| **Price per 1000 tokens** 	| **# of tokens** 	| **Price**        	|
| Claude 3   Sonnet             	| Slide/image   description         	| $0.0030                   	| 60              	| $0.000180    	| $0.0150                   	| 350             	| $0.005250        	|
| Titan Text   Embeddings       	| Slide/image description embedding 	| $0.0001                   	| 350             	| $0.000035    	| $0.0000                   	| 0               	| $0.000000        	|
| Titan Text   Embeddings       	| Question embedding                	| $0.0001                   	| 20              	| $0.000002    	| $0.0000                   	| 0               	| $0.000000        	|
| Claude 3   Sonnet             	| Final response                    	| $0.0030                   	| 700             	| $0.002100    	| $0.0150                   	| 8               	| $0.000120        	|
| Cost per   input/output       	|                                   	|                           	|                 	| $0.002317    	|                           	|                 	| $0.005370        	|
| **Total cost per   question** 	|                                   	|                           	|                 	|              	|                           	|                 	| **$0.007687** 	|



## Conclusion

In this series, we explored ways to use the power of multimodal FMs such as Titan
Multimodal Embeddings, Titan
Text Embeddings, and Claude 3 Sonnet models to discover new information
and uncover new perspectives on content in slide decks. We encourage you to explore different Claude models available on Bedrock.

With Generative AI being a fast moving space, there are several ways to improve the results and/or approach the problem. We are exploring performing a hybrid search and adding search filters by extracting entities from the question to improve the results. Lookout for a blog on "Talk to your PDF files (Enhanced Multimodal RAG) using foundation models (FMs) hosted on Amazon Bedrock and hybrid search" that will explore these concepts in detail.


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
