## About this repository

This is a 3-part series on the topic 'talk to your slide deck' using multimodal foundation models hosted on Amazon Bedrock and Amazon SageMaker. 

- [Part 1](https://github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog1-TitanEmbeddings-LVM) of the series implements a solution design that first converts slides into embeddings and stores these embeddings in a vector store (Amazon OpenSearch Serverless). When a user queries this data, LLaVA 1.5 is invoked and inference returned to user.
- [Part 2](https://github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog2-LVM-TitanEmbeddings) of the series follows a different solution design. This approach will generate and store LLaVA 1.5 inferences in a vector store (Amazon OpenSearch Serverless) and use those stored inferences to respond to user queries. 
- [Part 3](https://github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog3-Comparison-of-Approaches) of the series will compare the two approaches.
