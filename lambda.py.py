# lambda.py

# --- Lambda 1: serializeImageData ---

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    key = event['s3_key']
    bucket = event['s3_bucket']
    s3.download_file(bucket, key, "/tmp/image.png")
    
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())
    
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data.decode('utf-8'),
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

# --- Lambda 2: classifyImage---

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

ENDPOINT = "image-classification-2025-08-16-17-08-27-723"

def lambda_handler(event, context):
    image = base64.b64decode(event['body']['image_data'])
    
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=ENDPOINT,
        sagemaker_session=sagemaker.Session()
    )
    
    predictor.serializer = IdentitySerializer("image/png")
    inferences = predictor.predict(image)
    
    event['body']["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event['body'])
    }

# --- Lambda 3: filterConfidence---
import json
THRESHOLD = .93

def lambda_handler(event, context):
    # The input from the previous Lambda is a JSON string in the 'body', so we parse it.
    event_body = json.loads(event['body'])
    inferences = json.loads(event_body['inferences'])
    
    if max(inferences) < THRESHOLD:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event_body)
    }