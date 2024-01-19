# Databricks notebook source
# MAGIC %md
# MAGIC ### Take raw files from AZ Blob and convert them to text pre QA generation

# COMMAND ----------

#%pip install --upgrade azure-storage #not needed for 14.2
#%pip install azure-storage-blob --upgrade #not needed for 14.2
#%pip install "unstructured[docx,doc,pdf,pptx]"

from azure.core.credentials import AzureKeyCredential, AzureNamedKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean

from io import BytesIO
import os

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# COMMAND ----------

# MAGIC %md
# MAGIC Connect to container

# COMMAND ----------

container_client = ContainerClient(account_url = config['az_blob']['url'],
                                   credential=AzureNamedKeyCredential(name = config['az_blob']['name'],
                                                                      key = config['az_blob']['api']),
                                   container_name = 'oss-test')

# COMMAND ----------

blob_list = [i.name for i in container_client.list_blobs()]

print(f'Num of docs: {len(blob_list)}')
print(f'Unique extensions: {set([i.split(".")[-1] for i in blob_list])}')
print(f'Folders: {set([i.split("/")[0] for i in blob_list])}')

# COMMAND ----------

# MAGIC %md
# MAGIC Load files

# COMMAND ----------

category_mapper = {
    'cvs' : 'CV',
    'aa' : 'Advanced Analytics Reference document', 
    'knowledge' : 'Information about Hiflylabs'
    }

# COMMAND ----------

i = blob_list[16]
i

# COMMAND ----------

blob_downloaded = container_client.download_blob(i)

folder = os.path.dirname(i)
category = category_mapper[folder]
title = os.path.basename(i)

stream = BytesIO()
blob_downloaded.readinto(stream)

doc = partition_pdf(file = stream)
whole_doc = '\n'.join([i.text for i in doc])

print(f"Doc's title: {title}")
print(f"Doc's category: {category}")

# COMMAND ----------


