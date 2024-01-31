# Databricks notebook source
# MAGIC %md
# MAGIC ### Run SFT with prepared convo dataset

# COMMAND ----------

#%pip install -U git+https://github.com/huggingface/transformers
#%pip install -U git+https://github.com/huggingface/peft.git
#%pip install -U git+https://github.com/huggingface/accelerate.git
#%pip install trl bitsandbytes
#%pip install flash-attn --no-build-isolation
#%pip install py7zr

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk

import mlflow
import os
import shutil
import py7zr
from getpass import getpass

import torch
from multiprocessing import cpu_count

from transformers import BitsAndBytesConfig #for loading the base model in 4bits (instead of 32  or 16)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainingArguments

import yaml
with open('../config.yml', 'r') as file:
    config = yaml.safe_load(file)

from huggingface_hub import notebook_login, login
login(token = config['hf']['token'], write_permission=True)

# COMMAND ----------

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load SFT dataset

# COMMAND ----------

password = getpass()
archive_path = 'prepared_sft_data.7z'
op = 'prepared_sft_data.hf'

with py7zr.SevenZipFile(archive_path, mode='r', password=password) as z:
    z.extractall(path='.') #targets = [op]

dataset = load_from_disk(op)
#shutil.rmtree(op)

dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Train-test split

# COMMAND ----------

#https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/main_classes#datasets.Dataset.train_test_split.stratify_by_column

dataset_split = dataset.train_test_split(test_size = .05, shuffle = False, seed = 42)
dataset_split

# COMMAND ----------

train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

# COMMAND ----------

train_dataset

# COMMAND ----------

eval_dataset

# COMMAND ----------

dataset

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load HF elements

# COMMAND ----------

model_id = 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True, verbose = False)

# COMMAND ----------

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 1024

# set chat template

CUSTOM_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'kristof' %}\n{{ '<|kristof|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'timi' %}\n{{ '<|timi|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE

# COMMAND ----------

print(train_dataset[0]['text'])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up training settings
# MAGIC
# MAGIC Will use 4-bit quantization
# MAGIC - base model will be in 4bits
# MAGIC - adapter trainers and gradients will be bf16 (fp16, so half-precision)
# MAGIC - optimizer states remain 32bits

# COMMAND ----------

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    trust_remote_code=True
)
model.config.use_cache = False

# COMMAND ----------

# based on config
# https://huggingface.co/docs/peft/v0.7.1/en/package_reference/lora#peft.LoraConfig
peft_config = LoraConfig(
    r=16, # 64 with 128 alpha was too large; even 32 was too large to save...
    lora_alpha=32, #128
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L225 #gate_proj, up_proj, down_proj
)

# COMMAND ----------

model = get_peft_model(model, peft_config)

# COMMAND ----------

model.print_trainable_parameters()

# COMMAND ----------

# path where the Trainer will save its checkpoints and logs
output_model_path = 'mistral-fb-chat-sft-lora-r16-a32'
output_dir = f'data/{output_model_path}'

# based on https://huggingface.co/docs/transformers/main_classes/trainer
training_args = TrainingArguments(
    bf16=True, 
    do_eval=False, # with eval time would have taken 3-4 hours, even with 16 batch size
    evaluation_strategy="no", #epoch, steps
    gradient_accumulation_steps=4, #https://stackoverflow.com/questions/76002567/how-is-the-number-of-steps-calculated-in-huggingface-trainer
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4, #2.0e-05
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine", #linear, cosine
    optim="paged_adamw_32bit",
    max_steps=-1,
    num_train_epochs=3,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=8, # originally set to 8; 16 with 2 grad_acc was OOM; 8 + 4 is OK
    per_device_train_batch_size=8, # originally set to 8
    push_to_hub=True,
    hub_model_id=output_model_path,
    hub_strategy="every_save",
    hub_private_repo=True,
    report_to="mlflow",
    save_strategy="no", # if epoch: save checkpoints after each epoch
    save_total_limit=None,
    seed=42,
)

trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        #eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        #packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC MLFlow

# COMMAND ----------

#https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/callback#transformers.integrations.MLflowCallback
#https://gitlab.com/juliensimon/huggingface-demos/-/blob/main/mlflow/MLflow%20and%20Transformers.ipynb

os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/kristof.rabay@hiflylabs.com/oss-sft-mlflow-tracking"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"]="1"

# COMMAND ----------

# MAGIC %md
# MAGIC Train

# COMMAND ----------

train_result = trainer.train()

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC Save model

# COMMAND ----------

trainer.save_state()

# COMMAND ----------

trainer.save_model(output_dir)

# COMMAND ----------

!ls -lh -S data/mistral-fb-chat-sft-lora-r16-a32

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use for inference

# COMMAND ----------

def ResponseGenerator(user, message, model, generation_only = True):

    if user == 'kristof':
        responder = 'timi'
    
    elif user == 'timi':
        responder = 'kristof'

    formatted_message = f'<|{user}|>\n{message}</s>\n<|{responder}|>\n'

    # prepare the messages for the model
    # input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt", verbose = False).to("cuda")

    # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.encode
    input_ids = tokenizer.encode(formatted_message, add_special_tokens = False, return_tensors="pt").to('cuda')

    # for prompt len: https://colab.research.google.com/drive/1k6C_oJfEKUq0mtuWKisvoeMHxTcIxWRa?usp=sharing

    # inference
    outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=73,
            do_sample=True,
            temperature=0.1,
            eos_token_id= tokenizer.eos_token_id,
            top_k=50,
            top_p=0.95,
    )

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

    if generation_only:
        return output_text[len(input_text):]
    else:
        return output_text

# COMMAND ----------

# MAGIC %md
# MAGIC Try with current model, without reloading

# COMMAND ----------

question = 'Mit szeretnél holnap csinálni?'
user = 'kristof'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

question = 'Mit szeretnél holnap csinálni?'
user = 'timi'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

question = 'Milyen filmet nézzünk ma?'
user = 'kristof'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

user = 'timi'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

question = 'Hova utazzunk?'
user = 'kristof'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

user = 'timi'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

question = 'Fú, izgulok a holnapi nap miatt - mit csináljak?'
user = 'kristof'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------

user = 'timi'

answer = ResponseGenerator(user, question, trainer.model, generation_only = False)
print(answer)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Reloading

# COMMAND ----------

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(model = base_model, model_id  = output_dir)
ft_model = peft_model.merge_and_unload()

# COMMAND ----------

question = 'Hogy vagy?'
user = 'kristof'

answer = ResponseGenerator(user, question, ft_model, generation_only = False)
print(answer)
