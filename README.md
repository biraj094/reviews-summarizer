# Real-time Amazon Review Product Summarizer

## Overview
This project implements a real-time product review summarization system for Amazon products. It combines data engineering, machine learning, and MLOps principles to process, analyze, and visualize customer sentiment from large volumes of unstructured text data.

The system provides timely and structured insights from user-generated feedback, making it valuable for:
- Product teams
- Marketing departments
- E-commerce analysts

By leveraging AWS infrastructure and OpenAI's GPT-3.5, the project delivers an efficient, automated system that bridges data engineering with natural language processing.

## Architecture

### Data Engineering Pipeline
1. **Data Ingestion**
   - CSV files containing Amazon reviews are uploaded to S3 bucket
   - Files are named using product ASIN (e.g., B00LGEKOMS.csv)
   - AWS Lambda function triggers on file upload

2. **Data Processing**
   - Lambda performs data cleaning and timestamp validation
   - Reviews are processed in batches of 500
   - Titles and texts are concatenated for summarization
   - Metadata includes ASIN, batch number, and timestamp range
   - Processed batches sent to AWS Kinesis Data Stream

3. **Key Components**
   - AWS S3 for storage
   - AWS Lambda for serverless processing
   - AWS Kinesis Data Stream for real-time delivery
   - Pandas for preprocessing

### Machine Learning Model

The system uses OpenAI's GPT-3.5 model for abstractive summarization:
- Each batch of 500 reviews is processed
- Model generates structured summaries including:
  - Overall sentiment
  - Good points
  - Bad points
  - Sentiment score (1-10 Likert scale)
- Results stored as JSON objects with metadata

### MLOps Implementation

1. **Processing Pipeline**
   - Lambda function triggered by Kinesis stream records
   - OpenAI API processes review batches
   - Results stored in output S3 bucket

2. **Visualization**
   - Streamlit web application for results display
   - Interactive dashboard showing:
     - Available ASINs
     - Processed batches
     - Time-series sentiment graphs
     - Detailed summaries

## Performance Analysis

### Processing Metrics
- Large CSV processing: < 1 minute
- Typical product (6000 reviews): 12 batches
- Batch processing time: 1.5-2 seconds

### Cost Estimation (per 100 batches)
- AWS Lambda: ~$0.20
- AWS Kinesis: ~$0.015/hour/shard
- AWS S3: Negligible
- OpenAI API: ~$0.02/batch
- Total estimated cost: ~$2.25

## Future Work

1. **Monitoring & Alerts**
   - Implement drift detection
   - AWS SNS integration for alerts
   - DynamoDB/Redshift integration

2. **Infrastructure**
   - CI/CD pipelines (AWS CodePipeline/GitHub Actions)
   - Infrastructure as code (AWS SAM/Terraform)

## References

- [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)
- [Review Dataset 2023](https://amazon-reviews-2023.github.io/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [LLaMA Model](https://github.com/facebookresearch/llama)
- [OpenAI API](https://platform.openai.com/docs/)
- ROUGE Metrics: Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries."

## System Flow Diagram

```mermaid
flowchart TD
   A["S3 Bucket (Input)\nfinal-biraj/input/"] --> B["Lambda 1:\nCSV Preprocessing & Batch Creation"]
   B --> C["Kinesis Data Stream\n(ReviewDataStream)"]
   C --> D["Lambda 2:\nOpenAI Inference & S3 Storage"]
   D --> E["S3 Bucket (Output)\nfinal-biraj/review-output/"]
   E --> F["Streamlit App:\nSentiment Visualization Dashboard"]

   subgraph "Alerting"
       E --> G["SNS Alert System:\nDrift Notifications"]
   end
```

## Lambda Functions

### Preprocessing Lambda

```python
import boto3
import pandas as pd
import io
import json
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        s3 = boto3.client('s3')
        kinesis = boto3.client('kinesis')

        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        logger.info(f"Processing file {key} from bucket {bucket}")

        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))

        # Validate timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("CSV file must contain a 'timestamp' column")

        # Convert timestamp with error handling
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting timestamps: {str(e)}")
            raise

        # Drop rows with invalid timestamps
        invalid_timestamps = df['timestamp'].isna()
        if invalid_timestamps.any():
            logger.warning(f"Found {invalid_timestamps.sum()} rows with invalid timestamps. These will be dropped.")
            df = df.dropna(subset=['timestamp'])

        if len(df) == 0:
            logger.warning("No valid data remaining after timestamp validation")
            return {
                'statusCode': 200,
                'body': 'No valid data to process'
            }

        df.sort_values('timestamp', inplace=True)

        # Extract ASIN from the file name
        import os
        asin = os.path.splitext(os.path.basename(key))[0]
        logger.info(f"🧪 Extracted ASIN from filename: {asin}")

        batch_size = 500
        num_batches = len(df) // batch_size + int(len(df) % batch_size != 0)

        logger.info(f"Processing {len(df)} records in {num_batches} batches for ASIN {asin}")

        for i in range(num_batches):
            batch_df = df.iloc[i*batch_size : (i+1)*batch_size]
            min_ts = batch_df['timestamp'].min().isoformat()
            max_ts = batch_df['timestamp'].max().isoformat()
            reviews_combined = "\n".join(batch_df['title'].fillna('') + " " + batch_df['text'].fillna(''))

            payload = {
                "asin": asin,
                "batch_number": i + 1,
                "min_timestamp": min_ts,
                "max_timestamp": max_ts,
                "reviews_batch": reviews_combined
            }

            kinesis.put_record(
                StreamName='ReviewDataStream',
                Data=json.dumps(payload),
                PartitionKey=asin
            )

        return {
            'statusCode': 200,
            'body': f'Successfully processed {len(df)} records'
        }

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

```

### Inference Lambda

```python
import boto3
import json
import base64
import os
import time

# Defensive import for openai
try:
    import openai
    openai.api_key = os.environ['OPENAI_API_KEY']
except ImportError as e:
    openai = None
    print("⚠️ Warning: OpenAI module not loaded. Likely due to pydantic_core error.", e)

print("🔐 OPENAI_API_KEY found:", 'OPENAI_API_KEY' in os.environ)

s3 = boto3.client('s3')

def lambda_handler(event, context):
    for record in event['Records']:
        try:
            # Decode Kinesis record
            payload = base64.b64decode(record['kinesis']['data'])
            data = json.loads(payload)

            asin = data['asin']
            batch = data['batch_number']
            min_ts = data['min_timestamp']
            max_ts = data['max_timestamp']
            full_reviews_text = data['reviews_batch']

            # Trim if too long
            if len(full_reviews_text) > 12000:
                reviews_text = full_reviews_text[:12000]
                input_truncated = True
            else:
                reviews_text = full_reviews_text
                input_truncated = False

            # Prompt for OpenAI
            system_prompt = (
                "You are a product review summarizer. Summarize the following 500 customer reviews.\n"
                "Your response should include:\n"
                "- A short paragraph summarizing the overall sentiment.\n"
                "- A list of Good Points.\n"
                "- A list of Bad Points.\n"
                "- Give an overall product sentiment rating on a scale from 1 (very bad) to 10 (excellent).\n"
                "Here are the reviews:\n\n"
            )

            prompt = system_prompt + reviews_text

            # Call OpenAI safely
            summary = "No summary generated."
            if openai:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7
                    )
                    summary = response['choices'][0]['message']['content']
                except Exception as e:
                    print("⚠️ OpenAI call failed:", e)
            else:
                print("❌ Skipped OpenAI call due to import failure.")

            # Prepare output
            result = {
                "asin": asin,
                "batch_number": batch,
                "min_timestamp": min_ts,
                "max_timestamp": max_ts,
                "input_truncated": input_truncated,
                "summary": summary
            }

            # Save to S3
            timestamp = int(time.time())
            output_key = f"review-output/{asin}_batch_{batch}_{timestamp}.json"

            s3.put_object(
                Bucket="final-biraj",
                Key=output_key,
                Body=json.dumps(result),
                ContentType="application/json"
            )

            print(f"✅ Batch {batch} summary saved to s3://final-biraj/{output_key}")

        except Exception as e:
            print(f"❌ Error in processing batch: {e}")
            raise

```

### Streamlit Dashboard

Please take a look at <code>/eda/data/streamlit.py </code>



