# Databricks notebook source
# MAGIC %md
# MAGIC ### Pushing adapters to HF
# MAGIC
# MAGIC Will run experiments by pushing to private repos. DBX cluster used runs on CPU, so hopefully pushing does not require actual loading, just setting dir

# COMMAND ----------

#%pip install -U git+https://github.com/huggingface/transformers
#%pip install -U git+https://github.com/huggingface/peft.git
#%pip install bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

from huggingface_hub import notebook_login
notebook_login(token = config['hf']['token'], write_permission = True)

# COMMAND ----------

model_id = 'HuggingFaceH4/zephyr-7b-beta'
output_model_path = 'zephyr-hifly-7b-sft-lora'
output_dir = f'data/{output_model_path}'

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



# COMMAND ----------


