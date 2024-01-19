# Databricks notebook source
# MAGIC %md
# MAGIC ### Take raw files from AZ Blob and convert them to text pre QA generation

# COMMAND ----------

#%pip install --upgrade azure-storage
#%pip install azure-storage-blob --upgrade

from azure.core.credentials import AzureKeyCredential, AzureNamedKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# COMMAND ----------

container_client = ContainerClient(account_url = config['az_blob']['url'],
                                   credential=AzureNamedKeyCredential(name = config['az_blob']['name'],
                                                                      key = config['az_blob']['api']),
                                   container_name = 'oss-test')

# COMMAND ----------

blob_list = [i.name for i in container_client.list_blobs()]

print(f'Num of docs: {len(blob_list)}')
print(f'Unique extensions: {set([i.split(".")[-1] for i in blob_list])}')

# COMMAND ----------


