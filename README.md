# Using Amazon Bedrock & Rekoginition - Unleashing the Power of Image Comprehension
✨ In this project, we will be using the following key AWS Services Amazon Rekognition, Amazon Bedrock and Amazon Polly. Amazon Rekognition will be using detect_labels API call to analyze the image and to detect labels. Amazon Bedrock will be used to generate the summary from labels detected and Polly will be used to convert text to speech conversion.

===============        *********************      ==================         ******************
## Architectural Diagram:
🔀 Flow Diagram of the project illustrated below

![Image Comprehension_Arch_Diagram](https://github.com/user-attachments/assets/bdaa409b-9ec9-43fa-9764-d49b25408145)

 *********************    ===============    *****************     =============== 
## Pre-requisite Libraries Required:
⚙️ Below are the libraries required to be installed

```
python -m pip install scikit-image
python -m pip install langchain
python -m pip install --upgrade boto3
python -m pip install botocore
```
I.     **scikit-image:** Powerful library for image processing tasks in Python<br>
II.    **langchain:**  Focuses on building generative models and language<br>
III.   **boto3:**    AWS (Amazon Web Services) SDK for Python<br>
IV.    **botocore:** Core functionality for making API requests to AWS services<br>

===============        *********************      ==================         ******************
## Initialization of Amazon Bedrock and Amazon Polly services:
💡 Amazon Bedrock and Amazon Polly initialization

```
bedrock_client = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
)
```
```
polly = boto3.client('polly')
```
## Image Analyzer in Amazon Rekognition:
👨‍🌾 Rekognition will use internally the detect_labels api call and it will analyze the image and process the labels which are having confidence score of greater than 95

```
def imageAnalyzer(input_img):
    
    rek_client = boto3.client('rekognition')

    print(input_img)
    with open(input_img, 'rb') as image:
        response = rek_client.detect_labels(Image={'Bytes': image.read()})

    labels = response['Labels']
    print(f'Found {len(labels)} labels in the image:')
    label_names = ''
    for label in labels:
        name = label['Name']
        confidence = label['Confidence']
        #print(f'> Label "{name}" with confidence {confidence:.2f}')
        if confidence>95:
            print(name + "|" + str(confidence))
            label_names = label_names + name + ","

    return label_names
```
## Invoking the generative models using prompt template:
🔮 Based on the labels detected from Amazon Rekognition, we will be generating the human readable summary with generative models using detected labels as input
```
prompt_claude = """
Human:  Here are the comma seperated list of labels/objects seen in the image:
<labels>
{labels}
</labels>
Please provide a human readible and Understandable summary based on these labels
Assistant:
"""
```
```
prompt_template_for_summary_generate = PromptTemplate.from_template(prompt_claude)
prompt_data_for_summary_generate = prompt_template_for_summary_generate.format(labels=label_names)
print("prompt_data_for_summary_generate : ->" + prompt_data_for_summary_generate)
```
## Invoking the LLM based on user prompt and LLM Model type:
🌈 Will be invoking the claude language model based on user input prompt and LLM type
```
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
```
## Use the Amazon Polly to convert generated response to speech:
🗣️ Will be using Amazon Polly service to convert the generated response from generative language model text to speech

```
response = polly.synthesize_speech(
    OutputFormat='mp3',  # Choose the desired output format (e.g., mp3)
    Text=response_text,  # The text you want to convert to speech
    VoiceId='Joanna'     # Select a voice (e.g., Joanna)
)
```
*********************    ===============    *****************     =============== 

                                                       😊 Thank You 🙌




