##pre-requisite libraries required:
import boto3
import json
from langchain import PromptTemplate
import os
from skimage import io
import image_analyzer
polly = boto3.client('polly')



#Using boto3 library invoking the bedrock model:
bedrock_client = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
)


#inoking the LLM model:
def interactWithLLM(prompt,llm_type):

 if llm_type == 'titan':
  print("**THE LLM TYPE IS -->" + llm_type)

  parameters = {
   "maxTokenCount":512,
   "stopSequences":[],
   "temperature":0,
   "topP":0.9
  }
  body = json.dumps({"inputText": prompt, "textGenerationConfig": parameters})
  # modelId should be based on different providers
  modelId = "amazon.titan-tg1-large" 
  accept = "application/json"
  contentType = "application/json"

  response = bedrock_client.invoke_model(
   body=body, modelId=modelId, accept=accept, contentType=contentType
  )

  response_body = json.loads(response.get("body").read())

  response_text_titan = response_body.get("results")[0].get("outputText")

  return response_text_titan
 
 elif llm_type == 'claude':
  print("**THE LLM TYPE IS -->" + llm_type)
  body = json.dumps({"prompt": prompt,
                 "max_tokens_to_sample":300,
                 "temperature":1,
                 "top_k":250,
                 "top_p":0.999,
                 "stop_sequences":[]
                  }) 
  # modelId should be based on different providers
  modelId = 'anthropic.claude-v2' 
  accept = 'application/json'
  contentType = 'application/json'

  response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
  response_body = json.loads(response.get('body').read())

  response_text_claud = response_body.get('completion')

  return response_text_claud
 
 
 
#prompt template
prompt_claude = """
Human:  Here are the comma seperated list of labels/objects seen in the image:
<labels>
{labels}
</labels>
Please provide a human readible and Understandable summary based on these labels
Assistant:
"""

#calling the image rekognition api to detect the labels:
label_names = image_analyzer.imageAnalyzer('input_img.jpg')

prompt_template_for_summary_generate = PromptTemplate.from_template(prompt_claude)
prompt_data_for_summary_generate = prompt_template_for_summary_generate.format(labels=label_names)
print("prompt_data_for_summary_generate : ->" + prompt_data_for_summary_generate)

response_text = interactWithLLM(prompt_data_for_summary_generate,'claude')
print('response_text --- \n' + response_text)


# Use Amazon Polly to synthesize speech
response = polly.synthesize_speech(
    OutputFormat='mp3',  # Choose the desired output format (e.g., mp3)
    Text=response_text,           # The text you want to convert to speech
    VoiceId='Joanna'     # Select a voice (e.g., Joanna)
)

# Save the speech as an audio file
with open("output.mp3", "wb") as f:
    f.write(response['AudioStream'].read())