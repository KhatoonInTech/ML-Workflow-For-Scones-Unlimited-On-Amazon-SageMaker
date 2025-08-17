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


import os
import io
import boto3
import json
import base64
# import sagemaker
# from sagemaker.serializers import IdentitySerializer


# setting the  environment variables
ENDPOINT_NAME = 'image-classification-2025-08-16-17-08-27-723'
# # We will be using the AWS's lightweight runtime solution to invoke an endpoint.
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])
    
    # Make a prediction:
    predictor = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                    #   ContentType='image/png',
                                    ContentType='application/x-image',
                                      Body=image)
    
    # We return the data back to the Step Function    
    event["inferences"] = json.loads(predictor['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        # 'body': json.dumps(event)
        "body": {
            "image_data": event["body"]['image_data'],
            "s3_bucket": event["body"]['s3_bucket'],
            "s3_key": event["body"]['s3_key'],
            "inferences": event['inferences'],
       }
    }

# --- Lambda 3: filterConfidence---
import json
THRESHOLD = 0.93

def lambda_handler(event, context):
    # 'inferences' is nested inside 'body'
    body = event['body'] if "body" in event else event
    inferences = body['inferences']

    # Check confidence threshold
    if max(inferences) < THRESHOLD:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
