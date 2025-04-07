# reviews-summarizer

The Amazon Review Product Summarizer project has made substantial progress in both Data Engineering and Machine Learning phases as of April 07, 2025. In Data Engineering, the pipeline has been advanced to ingest, clean, and transform Amazon review data, enabling efficient retrieval of product-specific reviews from a refined dataset. Concurrently, the Machine Learning phase has successfully initiated the Prod pipeline by leveraging OpenAI’s GPT-4 for high-quality summarization and embeddings, demonstrating practical NLP application on a sample product with 509 reviews. These efforts align with the proposal’s vision of a scalable, cloud-based summarization system, setting a strong foundation for subsequent integration and deployment stages.

## Data Engineering Progress:

- Completed ingestion of raw Amazon review data from S3, filtering by category and product ID (e.g., Electronics).
- Cleaned dataset by reducing irrelevant categories and normalizing text, enhancing data quality.
- Transformed and aggregated reviews into a single text blob per product ID, enabling targeted summarization.

## Machine Learning Progress:

- Selected and implemented GPT-4 for Prod pipeline summarization, generating insights from 509 reviews.
- Produced embeddings with text-embedding-3-large, supporting future quantitative analysis.
- Achieved preliminary summarization with GPT-4, identifying sentiment, pros, cons, trends, and suggestions, with groundwork for Stage LLM (LLaMA-13B) deployment underway.


Data. Inside a folder /data add the following files 

- https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
- https://amazon-reviews-2023.github.io/index.html#for-user-reviews



env-reviews-summarizer


