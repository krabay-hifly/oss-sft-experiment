# Databricks notebook source
# MAGIC %md
# MAGIC ### Pushing adapters to HF
# MAGIC
# MAGIC Pushing a PEFT-model will only actually push the adapters! So upon inference, load the 
# MAGIC - base model (public HF model)
# MAGIC - adapters (your finetuned adapters in your public / private repo)
# MAGIC - tokenizer (preferably your tokenizer if you changed the settings; otherwise the tokenizer that belongs to the base model)
# MAGIC Then merge them with PEFTModel
# MAGIC
# MAGIC Do this if pushing to hub was not set during training (`TrainingArguments`). Setting it to private works only during training - if pushing afterwards, it will create a public repo, need to set it to private manually

# COMMAND ----------

#%pip install -U git+https://github.com/huggingface/transformers
#%pip install --upgrade huggingface_hub # for notebook_login()
#%pip install -U git+https://github.com/huggingface/peft.git
#%pip install -U git+https://github.com/huggingface/accelerate.git
#%pip install trl bitsandbytes
#%pip install flash-attn --no-build-isolation

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!nvidia-smi

# COMMAND ----------



# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

from huggingface_hub import notebook_login, login
login(token = config['hf']['token'], write_permission=True)

# COMMAND ----------

model_id = 'HuggingFaceH4/zephyr-7b-beta'

output_model_path = 'zephyr-hifly-7b-sft-lora'
output_dir = f'data/{output_model_path}'

hf_username = 'krabay'

# COMMAND ----------

# MAGIC %md
# MAGIC Push Adapters

# COMMAND ----------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code = True
)

# COMMAND ----------

model = PeftModel.from_pretrained(model, output_dir)

# COMMAND ----------

model.push_to_hub(f'{hf_username}/{output_model_path}', hub_private_repo=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Push tokenizer

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(output_dir)
tokenizer.push_to_hub(f'{hf_username}/{output_model_path}', hub_private_repo=True)
