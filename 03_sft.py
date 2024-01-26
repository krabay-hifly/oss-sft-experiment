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
# MAGIC - Worker type options: 
# MAGIC   - Standard_NC8as_T4_v3
# MAGIC     - 56GB RAM
# MAGIC     - 1 GPU (NVIDIA T4)
# MAGIC   - NC6_v3 
# MAGIC     - 112GB RAM
# MAGIC     - 1 GPU (V100) - https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series
# MAGIC   - A100
# MAGIC     - 220GB RAM
# MAGIC     - 1 GPU (A100) - https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series
# MAGIC
# MAGIC - `flash-attn-2` may not work on some GPUs, will either not use it, upgrade to newer GPU or downgrade package version
# MAGIC - also, T4 does not support `bf16`, so `fp16` will need to be set for anything related to quantization
# MAGIC
# MAGIC **Chosen infra**: A100 - should be no problems with flash attention, bf16, etc...

# COMMAND ----------

#%pip install -U git+https://github.com/huggingface/transformers
#%pip install -U git+https://github.com/huggingface/peft.git
#%pip install -U git+https://github.com/huggingface/accelerate.git
#%pip install trl bitsandbytes
#%pip install flash-attn --no-build-isolation

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict

import mlflow
import os

import torch
from multiprocessing import cpu_count

from transformers import BitsAndBytesConfig #for loading the base model in 4bits (instead of 32  or 16)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainingArguments

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

from huggingface_hub import notebook_login, login
login(token = config['hf']['token'], write_permission=True)

# COMMAND ----------

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# COMMAND ----------

data = pd.read_csv('data/QA_dataset.csv')
data.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load `tokenizer` from HuggingFace
# MAGIC
# MAGIC Model will be loaded later (on a GPU cluster)
# MAGIC
# MAGIC Possible models
# MAGIC - `teknium/OpenHermes-2.5-Mistral-7B` (https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
# MAGIC - `mistralai/Mistral-7B-v0.1`
# MAGIC - `HuggingFaceH4/zephyr-7b-beta`

# COMMAND ----------

# MAGIC %md
# MAGIC Using `chat_template` from HuggingFace, one can easily turn their data into the appropriate format

# COMMAND ----------

model_id = 'mistralai/Mistral-7B-v0.1'
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

split_index_list = data.sample(frac = 0.15, random_state = 21).index.tolist()
train = data[~data.index.isin(split_index_list)].reset_index(drop = True)
test = data[data.index.isin(split_index_list)].reset_index(drop = True)
                                
train_dataset = Dataset.from_pandas(train)
eval_dataset = Dataset.from_pandas(test)
whole_dataset = Dataset.from_pandas(data)

# COMMAND ----------

def apply_chat_template(example, tokenizer):

    message = example['messages']
    # We add an empty system message if there is none
    if message[0]["role"] != "system":
        message.insert(0, {"role": "system", "content": ""})
    example['text'] = tokenizer.apply_chat_template(message, tokenize=False, verbose = False)

    return example


column_names = list(train_dataset.features) #remove original columns, keep only text formatted for SFT

train_dataset = train_dataset.map(apply_chat_template,
                              num_proc=cpu_count(),
                              fn_kwargs={"tokenizer": tokenizer},
                              remove_columns=column_names,
                              desc="Applying chat template",)
eval_dataset = eval_dataset.map(apply_chat_template,
                              num_proc=cpu_count(),
                              fn_kwargs={"tokenizer": tokenizer},
                              remove_columns=column_names,
                              desc="Applying chat template",)
whole_dataset = whole_dataset.map(apply_chat_template,
                              num_proc=cpu_count(),
                              fn_kwargs={"tokenizer": tokenizer},
                              remove_columns=column_names,
                              desc="Applying chat template",)

# COMMAND ----------

print(train_dataset['text'][0])

# COMMAND ----------

whole_dataset

# COMMAND ----------

train_dataset

# COMMAND ----------

eval_dataset

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
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    #use_flash_attention_2=True,
    attn_implementation="flash_attention_2", # use it if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)

# COMMAND ----------

# path where the Trainer will save its checkpoints and logs
output_model_path = 'mistral-hifly-7b-sft-lora-r64-a128'
output_dir = f'data/{output_model_path}'

# COMMAND ----------

# based on https://huggingface.co/docs/transformers/main_classes/trainer
training_args = TrainingArguments(
    bf16=True, # specify bf16=True instead when training on GPUs that support bf16; default: fp16 = True
    do_eval=False, #True if there is an eval_dataset
    evaluation_strategy="no", #epoch, steps
    gradient_accumulation_steps=2, #128
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
    per_device_eval_batch_size=2, # originally set to 8
    per_device_train_batch_size=2, # originally set to 8
    push_to_hub=True,
    hub_model_id=output_model_path,
    hub_strategy="every_save",
    hub_private_repo=True,
    report_to="mlflow",
    save_strategy="no", # if epoch: save checkpoints after each epoch
    save_total_limit=None,
    seed=42,
)

# based on config
# https://huggingface.co/docs/peft/v0.7.1/en/package_reference/lora#peft.LoraConfig
peft_config = LoraConfig(
        r=64, 
        lora_alpha=128, 
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L225
)

trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=whole_dataset, #train_dataset
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

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# COMMAND ----------

train_result = trainer.train()

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC Save model

# COMMAND ----------

#metrics = train_result.metrics
#max_train_samples = training_args.max_train_samples if training_args.#max_train_samples is not None else len(train_dataset)
#max_train_samples = len(train_dataset)
#metrics["train_samples"] = min(max_train_samples, len(train_dataset)) 
#trainer.log_metrics("train", metrics)
#trainer.save_metrics("train", metrics)
trainer.save_state()

# COMMAND ----------

trainer.save_model(output_dir)

# COMMAND ----------

!ls -lh -S data/mistral-hifly-7b-sft-lora

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use for inference

# COMMAND ----------

# MAGIC %md
# MAGIC Try with current model, without reloading

# COMMAND ----------

def ResponseGenerator(question, model, generation_only = True):

    messages = [
        {"role": "system", "content": "You are the AI assistant of Hiflylabs, a Data & AI company. Your job is to answer employees' questions"},
        {"role": "user", "content": question}
    ]

    # prepare the messages for the model
    input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt", verbose = False).to("cuda")

    # for prompt len: https://colab.research.google.com/drive/1k6C_oJfEKUq0mtuWKisvoeMHxTcIxWRa?usp=sharing

    # inference
    outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.01,
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

question = "Who are some people whose CVs you recognize?"
answer = ResponseGenerator(question, trainer.model)
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

# after experimentation
# - once PeftModel.from_pretrained is called, base_model gets overwritten

# COMMAND ----------

question = "Who are some people whose CVs you recognize?"
answer = ResponseGenerator(question, ft_model) # so here ft_model = peft_model = base_model
print(answer)
