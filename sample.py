import json

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

body = json.dumps(
    {
        "prompt": "\n\nHuman:Write me a 100 word essay about snickers candy bars\n\nAssistant:",
        "max_tokens_to_sample": 200,
    }
).encode()

response = client.invoke_model_with_response_stream(
    body=body,
    modelId="anthropic.claude-v2",
    accept="application/json",
    contentType="application/json",
)

stream = response["body"]
if stream:
    for event in stream:
        chunk = event.get("chunk")
        if chunk:
            print(json.loads(chunk.get("bytes").decode()))