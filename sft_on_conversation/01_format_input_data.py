# Databricks notebook source
# MAGIC %md
# MAGIC ### Prepare facebook chat history for SFT
# MAGIC
# MAGIC Facebook conversation histroy downloaded from meta in JSON format, preprocessed locally, saved to CSV and that CSV is loaded here. Instead of instruction-completion and user-assistant tags, the tags here will be the two people conversing. Consecutive messages sent by the same person will be concatenated into 1 larger message. Extremely long messages will be dropped after looking at word count distribution

# COMMAND ----------

#%pip install py7zr
#%pip install trl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import py7zr
import os
from getpass import getpass

from transformers import AutoTokenizer
from datasets import Dataset

from trl.trainer import ConstantLengthDataset

# COMMAND ----------

model_id = 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True, verbose = False)

# COMMAND ----------

# set pad_token_id equal to the eos_token_id if not set
#if tokenizer.pad_token_id is None:
#  tokenizer.pad_token_id = tokenizer.eos_token_id

# set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 1024

# set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

CUSTOM_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'kristof' %}\n{{ '<|kristof|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'timi' %}\n{{ '<|timi|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE

messages = [
    {"role": "timi", "content": "Szia :)"},
    {"role": "kristof", "content": "Hali, mizu?"},
    {"role": "timi", "content": "Hétvégén milyen filmet nézzünk?"},
    {"role": "kristof", "content": "Keresztapa 2-t azóta sem láttuk..."}
]

encodeds = tokenizer.apply_chat_template(messages, tokenize=False) #return_tensors="pt", add_generation_prompt=True 

print('Applied chat template to messages dict')
print('-'*80)
#print(tokenizer.decode(*encodeds))
print(encodeds)

# COMMAND ----------

# MAGIC %md
# MAGIC Load dataset

# COMMAND ----------

password = getpass()
archive_path = 'data_w_sentiment.7z'
output_path = 'data_w_sentiment.csv'

with py7zr.SevenZipFile(archive_path, mode='r', password=password) as z:
    z.extract(path='.', targets = [output_path])

data = pd.read_csv(output_path)
os.remove(output_path)

data.head(3)

# COMMAND ----------

data['num_words'].hist(bins = 50)

# COMMAND ----------

# filter out messages with more than 75 words
print(data.shape)
data = data[data['num_words'] <= 75]
print(data.shape)

# COMMAND ----------

data['content'] = data['content'].astype(str)
data['group'] =(data['sender_name'] != data.shift().fillna(method='bfill')['sender_name']).cumsum()

data_concat = data.groupby(['sender_name', 'group'], as_index = False).agg({'content': '. '.join}).sort_values('group').drop('group', axis = 1).reset_index(drop = True)
data_concat.head(3)


# COMMAND ----------

# MAGIC %md
# MAGIC Check chunk-word cound distribution again (after concatting multiple messages into 1)

# COMMAND ----------

word_count_in_chunks = data_concat['content'].apply(lambda x: len(x.split())).tolist()
plt.hist(word_count_in_chunks, bins = 50)
plt.show()

# COMMAND ----------

# filter messages with more than 100 words

data_concat['num_words'] = data_concat['content'].apply(lambda x: len(x.split()))

print(data_concat.shape)
data_concat = data_concat[data_concat['num_words'] < 100].reset_index(drop = True)
print(data_concat.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create SFT dataset

# COMMAND ----------

remap_names = {'Tímea Hunka' : 'timi', 'Kristóf Rábay' : 'kristof'}
data_concat['sender_name'] = data_concat['sender_name'].map(remap_names)

# COMMAND ----------

data_concat.head(6)

# COMMAND ----------

max_index = max(data_concat.index)
convos = []

for index, row in tqdm(data_concat.iterrows()):
    if index % 2 == 0 and index < max_index:

        current_row = data_concat.loc[index]
        next_row = data_concat.loc[index+1]

        convo = [{'role' : current_row['sender_name'], 'content' : current_row['content']},
                 {'role' : next_row['sender_name'], 'content' : next_row['content']}]
        
        convos.append(convo)

    else:
        pass


# COMMAND ----------

print(len(convos))

# COMMAND ----------

for i in convos[:4]:
    print(i)
    print('--'*80)

# COMMAND ----------

convos_applied_template = []

for i in tqdm(convos):
    convos_applied_template.append(tokenizer.apply_chat_template(i, tokenize=False, verbose = False))
    
print(len(convos))
print(len(convos_applied_template))

# COMMAND ----------

for i in convos_applied_template[:4]:
    print(i)
    print('--'*80)

# COMMAND ----------

df = pd.DataFrame(convos_applied_template, columns =  ['text'])
sft_dataset = Dataset.from_pandas(df)

sft_dataset

# COMMAND ----------

sft_dataset_constant_length = ConstantLengthDataset(dataset = sft_dataset, tokenizer = tokenizer, dataset_text_field = 'text')
sft_dataset_constant_length_elements = [i for i in sft_dataset_constant_length]
print(len(sft_dataset_constant_length_elements))

# COMMAND ----------

# MAGIC %md
# MAGIC Generated dataset is fine, but maybe manual generation with randomized lengths is more supervised - will revisit this
