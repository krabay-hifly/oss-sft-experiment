# Databricks notebook source
# MAGIC %md
# MAGIC ### Run SFT with prepared QA dataset
# MAGIC
# MAGIC Example notebooks / materials:
# MAGIC - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
# MAGIC
# MAGIC Infra
# MAGIC - DBX ML Runtime: 14.2.x-gpu-ml-scala2.12
# MAGIC - Worker type: Standard_NC8as_T4_v3
# MAGIC   - 56GB RAM
# MAGIC   - 1 GPU (NVIDIA T4)

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from tqdm import tqdm
import time

import json
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# COMMAND ----------

data = pd.read_csv('data/QA_dataset.csv')
data.head(2)

# COMMAND ----------


