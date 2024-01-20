# Databricks notebook source
# MAGIC %md
# MAGIC ### Take chunks and create QA dataset for SFT

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import openai
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import json
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name=config['az_oai']['deployment']

# COMMAND ----------

def generate_response(messages, temperature = 0.0):

    completion = openai.ChatCompletion.create(
        engine=deployment_name, 
        messages=messages, 
        temperature=temperature)
    
    response = completion.choices[0]['message']['content']
    usage = completion.usage
    return response, usage
    
prompt = 'Write a tagline for an ice cream shop.'
messages = [{'role' : 'user', 'content' : prompt}]

response, usage = generate_response(messages)
print(usage)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC Check chunk-word cound distribution, filter low values

# COMMAND ----------

with open('data/chunks.json') as json_file:
    chunks = json.load(json_file)

chunks[65]

# COMMAND ----------

word_count_in_chunks = [len(i['content'].split()) for i in chunks]

plt.hist(word_count_in_chunks, bins = 30)
plt.show()

# COMMAND ----------

# filter chunks with less than 20 words

print(len(chunks))
chunks = [i for i in chunks if len(i['content'].split()) > 20]
print(len(chunks))

# COMMAND ----------


