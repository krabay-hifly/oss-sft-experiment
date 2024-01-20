# Databricks notebook source
# MAGIC %md
# MAGIC ### Take chunks and create QA dataset for SFT

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import json

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

with open('data/chunks.json') as json_file:
    chunks = json.load(json_file)

chunks[65]

# COMMAND ----------


