# Databricks notebook source
# MAGIC %md
# MAGIC ### Run SFT with prepared QA dataset
# MAGIC
# MAGIC Example notebooks / materials:
# MAGIC - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
# MAGIC - https://gist.github.com/younesbelkada/f48af54c74ba6a39a7ae4fd777e72fe8
# MAGIC - https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb
# MAGIC - https://medium.com/@sujathamudadla1213/difference-between-trainer-class-and-sfttrainer-supervised-fine-tuning-trainer-in-hugging-face-d295344d73f7
# MAGIC - https://huggingface.co/blog/4bit-transformers-bitsandbytes
# MAGIC
# MAGIC Infra
# MAGIC - DBX ML Runtime: 14.2.x-gpu-ml-scala2.12
# MAGIC - Worker type: Standard_NC8as_T4_v3
# MAGIC   - 56GB RAM
# MAGIC   - 1 GPU (NVIDIA T4)
# MAGIC
# MAGIC - `flash-attn-2` may not work on this GPU, will either not use it, upgrade to newer GPU or downgrade package version
# MAGIC - also, T4 does not support `bf16`, so `fp16` will need to be set for anything related to quantization

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from multiprocessing import cpu_count

# COMMAND ----------

data = pd.read_csv('data/QA_dataset.csv')
data.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load `tokenizer` from HuggingFace
# MAGIC
# MAGIC Model will be loaded later (on a GPU cluster)
# MAGIC
# MAGIC Chosen model: `teknium/OpenHermes-2.5-Mistral-7B` (https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
# MAGIC
# MAGIC For tokenization using: `HuggingFaceH4/zephyr-7b-beta` as `teknium` isn't adding eos token

# COMMAND ----------

default_system_message = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
"""

default_conversation_template = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by a man named Teknium, who designed me to assist and support users with their needs and requests.<|im_end|>
"""

# COMMAND ----------

# MAGIC %md
# MAGIC Using `chat_template` from HuggingFace, one can easily turn their data into the appropriate format

# COMMAND ----------

model_id = 'HuggingFaceH4/zephyr-7b-beta'
tokenizer = AutoTokenizer.from_pretrained(model_id, verbose = False)

# COMMAND ----------

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 2048

# set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

messages = [
    {"role": "system", "content": "You are the AI assistant of Hiflylabs, a Data & AI company. Your job is to answer employees' questions"},
    {"role": "user", "content": "Hello, who are you?"},
    {"role": "assistant", "content": "Hi, I'm good, here to help you answer questions about Hiflylabs"},
    {"role": "user", "content": "What are the 12 points?"}
]

encodeds = tokenizer.apply_chat_template(messages, tokenize=False) #return_tensors="pt", add_generation_prompt=True 


print('Applied chat template to messages dict')
print('-'*80)
#print(tokenizer.decode(*encodeds))
print(encodeds)

# COMMAND ----------

# MAGIC %md
# MAGIC Apply to dataset

# COMMAND ----------

system_message = "You are the AI assistant of Hiflylabs, a Data & AI company. Your job is to answer employees' questions."

data['messages'] = data.apply(lambda x: [{'role': 'system', 'content': system_message},
                                         {'role': 'user', 'content': x['questions_split']}, 
                                         {'role': 'assistant', 'content': x['answers']}], axis = 1)
                                         
sft_dataset = Dataset.from_pandas(data)

# COMMAND ----------

def apply_chat_template(example, tokenizer):

    message = example['messages']
    # We add an empty system message if there is none
    if message[0]["role"] != "system":
        message.insert(0, {"role": "system", "content": ""})
    example['text'] = tokenizer.apply_chat_template(message, tokenize=False, verbose = False)

    return example


column_names = list(sft_dataset.features) #remove original columns, keep only text formatted for SFT
sft_dataset = sft_dataset.map(apply_chat_template,
                              num_proc=cpu_count(),
                              fn_kwargs={"tokenizer": tokenizer},
                              remove_columns=column_names,
                              desc="Applying chat template",)

# COMMAND ----------

print(sft_dataset['text'][0])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# for later
#device = 'cuda'

#model = AutoModelForCausalLM.from_pretrained(model_id)

#model_inputs = encodeds.to(device)
#model.to(device)

#generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
#decoded = tokenizer.batch_decode(generated_ids)
#print(decoded[0])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Load model from `HuggingFace`

# COMMAND ----------



# COMMAND ----------


