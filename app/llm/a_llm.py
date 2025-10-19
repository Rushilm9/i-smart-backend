import os
import json
from dotenv import load_dotenv
import boto3

# Load .env
load_dotenv(override=True)

region = os.getenv("AWS_REGION", "us-east-1")
model_id = os.getenv("BEDROCK_MODEL_PROFILE_ARN")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not model_id:
    raise ValueError("No model ID found. Check your .env file.")

# Initialize Bedrock client
client = boto3.client(
    "bedrock-runtime",
    region_name=region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Prompt to send
prompt = "what is teh capital og delhi"

# Body for invoke-model API (matches your CLI JSON)
body = {
    "messages": [
        {"role": "user", "content": [{"text": prompt}]}
    ],
    "inferenceConfig": {
        "maxTokens": 512,
        "stopSequences": [],
        "temperature": 0.7,
        "topP": 0.9
    }
}

# Call model
response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(body)
)

# Read output text
result = json.loads(response["body"].read())
text = result["output"]["message"]["content"][0]["text"]

print("Model output:\n", text)
