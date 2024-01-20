# Databricks notebook source
# MAGIC %md
# MAGIC ### Take raw files from AZ Blob and convert them to text pre QA generation

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

#%pip install --upgrade azure-storage #not needed for 14.2
#%pip install azure-storage-blob --upgrade #not needed for 14.2
#%pip install "unstructured[docx,doc,pdf,pptx]"
#needed to install libreoffice: https://gcore.com/learning/how-to-install-libri-office-on-ubuntu/
#apt-get install -y poppler-utils 

import warnings
warnings.filterwarnings('ignore')

from azure.core.credentials import AzureKeyCredential, AzureNamedKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.doc import partition_doc
from unstructured.partition.pptx import partition_pptx

from unstructured.cleaners.core import clean

from tqdm import tqdm
from io import BytesIO
import os
import json

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

def partition_by_extension(ext, stream):

    ext = ext.lower()

    if ext == 'pdf':
        return partition_pdf(file = stream)
    if ext == 'docx':
        return partition_docx(file = stream)
    if ext == 'pptx':
        return partition_pptx(file = stream)

# COMMAND ----------

def load_data_to_large_chunks(blob_name):
    
    blob_downloaded = container_client.download_blob(blob_name)

    folder = os.path.dirname(blob_name)
    category = category_mapper[folder]
    title = os.path.basename(blob_name)
    extension = title.split('.')[-1]

    stream = BytesIO()
    blob_downloaded.readinto(stream)
    doc = partition_by_extension(extension, stream)

    chunks_from_each_page = []

    if doc[-1].metadata.page_number and category != 'CV': #even is multiple pages, CVs should be kept as 1 doc

        for page_num in range(doc[-1].metadata.page_number):
            page_num = page_num+1
            whole_doc = '\n'.join([i.text for i in doc if i.metadata.page_number == page_num])

            chunk = {
                        'title' : title,
                        'page' : page_num,
                        'category' : category,
                        'content' : whole_doc
                        }
            
            chunks_from_each_page.append(chunk)

    else:    

        whole_doc = '\n'.join([i.text for i in doc])

        chunk = {
            'title' : title,
            'page' : 1,
            'category' : category,
            'content' : whole_doc
            }
        
        chunks_from_each_page.append(chunk)

    return chunks_from_each_page

# COMMAND ----------

# MAGIC %%time
# MAGIC
# MAGIC all_chunks = []
# MAGIC errors = []
# MAGIC
# MAGIC for blob_name in tqdm(blob_list):
# MAGIC     try:
# MAGIC         chunk = load_data_to_large_chunks(blob_name)
# MAGIC         all_chunks.append(chunk)
# MAGIC     except:
# MAGIC         errors.append(blob_name)

# COMMAND ----------

print(f"Num docs throwing error: {len(errors)}")
print(f"Num docs successfully processed: {len(all_chunks)}")

# COMMAND ----------

all_chunks = sum(all_chunks, [])
print(f"Num docs after flattening list: {len(all_chunks)}")

# COMMAND ----------

with open('data/chunks.json', 'w') as fout:
    json.dump(all_chunks, fout)

# COMMAND ----------

# to read
#with open('data/chunks.json') as json_file:
#    test_read = json.load(json_file)

# COMMAND ----------


