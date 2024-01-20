# Databricks notebook source
# MAGIC %md
# MAGIC ### Run SFT with prepared QA dataset
# MAGIC
# MAGIC Example notebooks / materials:
# MAGIC - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb

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


