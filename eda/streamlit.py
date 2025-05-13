import streamlit as st
import pandas as pd
from datetime import datetime
import boto3
import json
import os
from dotenv import load_dotenv
import re
import plotly.express as px

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Review Analysis Dashboard",
    layout="wide"
)

# AWS Configuration from environment variables
BUCKET_NAME = os.getenv('S3_BUCKET', 'final-biraj')
PREFIX = os.getenv('S3_PREFIX', 'review-output').rstrip('/')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

@st.cache_resource
def get_s3_client():
    """Create and cache the S3 client connection"""
    try:
        credentials = {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'aws_session_token': os.getenv('AWS_SESSION_TOKEN'),
            'region_name': AWS_REGION
        }
        credentials = {k: v for k, v in credentials.items() if v is not None}
        return boto3.client('s3', **credentials)
    except Exception as e:
        st.error(f"Failed to connect to AWS: {str(e)}")
        return None

def list_all_files():
    """List all JSON files in the S3 bucket"""
    try:
        s3_client = get_s3_client()
        if not s3_client:
            return []
        
        prefix = f"{PREFIX}/" if not PREFIX.endswith('/') else PREFIX
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        if 'Contents' not in response:
            return []
            
        return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]
    except Exception as e:
        st.error(f"Error listing files from S3: {str(e)}")
        return []

def load_summary(key):
    """Load a specific summary file from S3"""
    try:
        s3_client = get_s3_client()
        if not s3_client:
            return {}
            
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        st.error(f"Error loading summary from S3 ({key}): {str(e)}")
        return {}

def format_timestamp(timestamp_str):
    if not timestamp_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y %H:%M")
    except:
        return timestamp_str

def parse_timestamp(timestamp_str):
    """Convert timestamp string to datetime object"""
    if not timestamp_str:
        return None
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except:
        return None

def extract_sentiment_score(summary_text):
    """Extract sentiment score from summary text"""
    if not summary_text:
        return None
    
    patterns = [
        r"rating:?\s*(\d+)(?:/10)?",
        r"rating\s+of\s+(\d+)(?:/10)?",
        r"average\s+rating\s*:?\s*(\d+)(?:/10)?",
        r"sentiment\s+score\s*:?\s*(\d+)(?:/10)?"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, summary_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 10)  # Ensure score is between 1 and 10
    
    return None

def group_files_by_product(files):
    """Group files by product prefix"""
    products = {}
    for file in files:
        match = re.search(r'review-output/([^_]+)_batch_', file)
        if match:
            product_id = match.group(1)
            if product_id not in products:
                products[product_id] = []
            products[product_id].append(file)
    
    return products

# Main app
st.title("Review Analysis Dashboard")

# Check AWS credentials
required_credentials = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
missing_credentials = [cred for cred in required_credentials if not os.getenv(cred)]

if missing_credentials:
    st.error(f"Missing AWS credentials: {', '.join(missing_credentials)}")
    st.info("Please ensure all required AWS credentials are set in environment variables")
    st.stop()

# Display AWS configuration
st.sidebar.subheader("AWS Configuration")
st.sidebar.text(f"Region: {AWS_REGION}")
st.sidebar.text(f"Bucket: {BUCKET_NAME}")
st.sidebar.text(f"Prefix: {PREFIX}")
st.sidebar.text("Credentials: ✓ Loaded")

try:
    # Get all files from S3
    all_files = list_all_files()
    if not all_files:
        st.warning("No files found in S3 bucket")
        st.stop()

    # Group files by product
    products = group_files_by_product(all_files)
    
    # Display total metrics
    total_products = len(products)
    total_batches = sum(len(files) for files in products.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Products", total_products)
    with col2:
        st.metric("Total Batches", total_batches)

    # Display products and their batches
    st.subheader("Products and Batches")
    
    for product_id, files in products.items():
        with st.expander(f"Product: {product_id}", expanded=False):
            # Load summaries for this product
            product_summaries = []
            for file_key in sorted(files):
                summary = load_summary(file_key)
                if summary:
                    # Add sentiment score and timestamp
                    summary['sentiment_score'] = extract_sentiment_score(summary.get('summary', ''))
                    summary['timestamp'] = parse_timestamp(summary.get('min_timestamp'))
                    product_summaries.append(summary)
            
            if product_summaries:
                # Create DataFrame for visualization
                df = pd.DataFrame(product_summaries)
                if not df.empty and 'batch_number' in df.columns:
                    df['batch_number'] = pd.to_numeric(df['batch_number'], errors='coerce')
                    df = df.sort_values('batch_number')
                    
                    # Display sentiment trend if scores are available
                    if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
                        st.subheader("Sentiment Trend")
                        
                        # Create time series plot with Plotly
                        fig = px.line(df, 
                                    x='timestamp', 
                                    y='sentiment_score',
                                    title='Sentiment Score Over Time',
                                    labels={'timestamp': 'Time', 'sentiment_score': 'Sentiment Score (1-10)'},
                                    range_y=[0, 10])
                        
                        # Customize the layout
                        fig.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Sentiment Score",
                            yaxis=dict(range=[0, 10], tickmode='linear', tick0=0, dtick=1),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate average sentiment
                        avg_sentiment = df['sentiment_score'].mean()
                        st.metric("Average Sentiment", f"{avg_sentiment:.1f}/10")
                    
                    # Display total reviews for this product
                    total_product_reviews = df['review_count'].sum() if 'review_count' in df.columns else 0
                    st.metric("Total Reviews", total_product_reviews)
                    
                    # Display all batch summaries in a table
                    st.subheader("Batch Summary Table")
                    summary_table = df[[
                        'batch_number', 
                        'timestamp', 
                        'sentiment_score', 
                        'review_count'
                    ]].copy() if all(col in df.columns for col in ['batch_number', 'timestamp', 'sentiment_score', 'review_count']) else pd.DataFrame()
                    
                    if not summary_table.empty:
                        summary_table['timestamp'] = summary_table['timestamp'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notnull(x) else "Unknown")
                        summary_table.columns = ['Batch Number', 'Timestamp', 'Sentiment Score', 'Reviews Count']
                        st.dataframe(summary_table, use_container_width=True)
                    
                    # Display individual batches using tabs
                    st.subheader("Detailed Batch Information")
                    tabs = st.tabs([f"Batch {summary.get('batch_number', 'Unknown')}" for summary in product_summaries])
                    for tab, summary in zip(tabs, product_summaries):
                        with tab:
                            st.write(f"Time Range: {format_timestamp(summary.get('min_timestamp'))} → {format_timestamp(summary.get('max_timestamp'))}")
                            if 'review_count' in summary:
                                st.write(f"Reviews in batch: {summary['review_count']}")
                            if summary.get('sentiment_score'):
                                st.write(f"Sentiment Score: {summary['sentiment_score']}/10")
                            st.info(summary.get('summary', 'No summary available'))

except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.exception(e)  # This will show the full error traceback

st.markdown("---")
st.caption("Review Analysis Dashboard")
