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

        # Validate ASIN column
        if 'asin' not in df.columns or df['asin'].iloc[0] is None:
            raise ValueError("CSV file must contain a valid 'asin' column")

        asin = str(df['asin'].iloc[0])
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
