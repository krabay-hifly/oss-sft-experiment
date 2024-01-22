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
# MAGIC - Worker type: Standard_NC8as_T4_v3
# MAGIC   - 56GB RAM
# MAGIC   - 1 GPU (NVIDIA T4)
# MAGIC
# MAGIC - `flash-attn-2` may not work on this GPU, will either not use it, upgrade to newer GPU or downgrade package version
# MAGIC - also, T4 does not support `bf16`, so `fp16` will need to be set for anything related to quantization

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict

import torch
from multiprocessing import cpu_count

from transformers import BitsAndBytesConfig #for loading the base model in 4bits (instead of 32  or 16)
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

# COMMAND ----------

data = pd.read_csv('data/QA_dataset.csv')
data.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load `tokenizer` from HuggingFace
# MAGIC
# MAGIC Model will be loaded later (on a GPU cluster)
# MAGIC
# MAGIC Chosen model: `teknium/OpenHermes-2.5-Mistral-7B` (https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
# MAGIC
# MAGIC For tokenization using: `HuggingFaceH4/zephyr-7b-beta` as `teknium` isn't adding eos token

# COMMAND ----------

default_system_message = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
"""

default_conversation_template = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by a man named Teknium, who designed me to assist and support users with their needs and requests.<|im_end|>
"""

# COMMAND ----------

# MAGIC %md
# MAGIC Using `chat_template` from HuggingFace, one can easily turn their data into the appropriate format

# COMMAND ----------

model_id = 'HuggingFaceH4/zephyr-7b-beta'
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
                                         
sft_dataset = Dataset.from_pandas(data)

# COMMAND ----------

def apply_chat_template(example, tokenizer):

    message = example['messages']
    # We add an empty system message if there is none
    if message[0]["role"] != "system":
        message.insert(0, {"role": "system", "content": ""})
    example['text'] = tokenizer.apply_chat_template(message, tokenize=False, verbose = False)

    return example


column_names = list(sft_dataset.features) #remove original columns, keep only text formatted for SFT
sft_dataset = sft_dataset.map(apply_chat_template,
                              num_proc=cpu_count(),
                              fn_kwargs={"tokenizer": tokenizer},
                              remove_columns=column_names,
                              desc="Applying chat template",)

# COMMAND ----------

print(sft_dataset['text'][0])

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
            bnb_4bit_compute_dtype="torch.bfloat16",
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    attn_implementation="flash_attention_2", # use it if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)

# COMMAND ----------

# path where the Trainer will save its checkpoints and logs
output_model_path = 'mistral-hifly-7b-sft-lora'
output_dir = f'data/{output_model_path}'

# based on config
training_args = TrainingArguments(
    fp16=True, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1, # originally set to 8
    per_device_train_batch_size=1, # originally set to 8
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    # report_to="tensorboard",
    save_strategy="no",
    save_total_limit=None,
    seed=42,
)

# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=sft_dataset,
        #train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Train

# COMMAND ----------

train_result = trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC Save model

# COMMAND ----------

train_result.metrics

# COMMAND ----------

metrics = train_result.metrics
max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(sft_dataset)) #used to be train_dataset
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Use for inference

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")

# COMMAND ----------

# We use the tokenizer's chat template to format each message
messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

# prepare the messages for the model
input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# inference
outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

# COMMAND ----------


