# Databricks notebook source
# MAGIC %md
# MAGIC ### Take chunks and create QA dataset for SFT
# MAGIC
# MAGIC Example notebooks / materials:
# MAGIC - https://docs.llamaindex.ai/en/stable/examples/finetuning/knowledge/finetune_knowledge.html
# MAGIC - https://arxiv.org/pdf/2401.08406.pdf
# MAGIC - https://www.pinecone.io/blog/rag-study/

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import openai
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
from collections import Counter
from functools import reduce
from operator import add

import json
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name_gpt_4_turbo = config['az_oai']['deployment_4']
deployment_name_gpt_35_turbo = config['az_oai']['deployment_35']

# COMMAND ----------

def generate_response(messages, deployment_name = deployment_name_gpt_4_turbo, temperature = 0.0):

    completion = openai.ChatCompletion.create(
        engine=deployment_name, 
        messages=messages, 
        temperature=temperature)
    
    response = completion.choices[0]['message']['content']
    usage = completion.usage.to_dict()
    return response, usage
    
prompt = 'Write a tagline for an ice cream shop.'
messages = [{'role' : 'user', 'content' : prompt}]

print('GPT-4')
response, usage = generate_response(messages)
print(usage)
print(response)

print('-'*100)
print('GPT-3.5')
response, usage = generate_response(messages, deployment_name=deployment_name_gpt_35_turbo)
print(usage)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC Check chunk-word cound distribution, filter low values

# COMMAND ----------

with open('data/chunks.json') as json_file:
    chunks = json.load(json_file)

chunks[65]

# COMMAND ----------

word_count_in_chunks = [len(i['content'].split()) for i in chunks]

plt.hist(word_count_in_chunks, bins = 30)
plt.show()

# COMMAND ----------

# filter chunks with less than 20 words

print(len(chunks))
chunks = [i for i in chunks if len(i['content'].split()) > 20]
print(len(chunks))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Put together prompt for Question generation

# COMMAND ----------

def QuestionGenerator(context, category, title, num_questions_per_chunk):

    system_message = "You are a Teacher / Professor. Your task is to setup a quiz / examination to assess the knowledge of a student about given context"

    instuction = f"""
    Using the provided context, formulate {num_questions_per_chunk} questions that capture important facts from the context.
    You must obey the following criteria:
    - Restrict the question to the context information
    - Do not create questions that cannot be answered from the context
    - Phrase the question so that it does not refer to specific context. For example, do NOT put phrases like 'given provided context' or 'in this work' in the question, because when the question is asked elsewhere it wouldn't be provided specific context. Replace these terms with specific details.

    BAD questions:
    - What did the author do in his childhood?
    - What were the main findings in this report?

    GOOD questions:
    - What did Obama do in his childhood?
    - What were the main finding of the original Transformer paper by Vaswani et al?

    Here is the context:
    {context}

    The document is a {category}, it's title is {title}.

    Generate the questions below:
    """

    messages = [{'role' : 'system', 'content' : system_message},
                {'role' : 'user', 'content' : instuction}]

    response, usage = generate_response(messages)

    return response, usage

# COMMAND ----------

# MAGIC %md
# MAGIC Approximate cost pre-generation

# COMMAND ----------

avg_input_tokens = 1000
input_cost = 0.01 / 1000
avg_output_tokens = 150
output_cost = 0.03 / 1000

avg_num_questions_per_chunk = 7

cost_approximation = len(chunks) * avg_num_questions_per_chunk * (avg_input_tokens * input_cost + avg_output_tokens * output_cost)
print(f'Approx cost in USD to generate questions: {cost_approximation}')

# COMMAND ----------

# MAGIC %md
# MAGIC Tie question number to content length

# COMMAND ----------

content_length_intervals = [range(100), range(100, 250), range(250, 500), range(500, max(word_count_in_chunks)*100) ]
question_number = [2, 5, 10, 15]

questions_generated = []

for i in tqdm(chunks):
    
    context = i['content']
    category = i['category']
    title = i['title']

    word_count = len(i['content'].split())
    range_lookup = [word_count in i for i in content_length_intervals]
    num_questions_per_chunk = question_number[range_lookup.index(max(range_lookup))]

    questions, usage = QuestionGenerator(context, category, title, num_questions_per_chunk)

    output = {**i, 
              'questions' : questions,
              'questions_number' : num_questions_per_chunk,
              'question_generation_token_usage' : usage}
    questions_generated.append(output)

    # sleep 2 secs after each API call
    time.sleep(2)

# COMMAND ----------

print(len(questions_generated))

#with open('data/questions.json', 'w') as fout:
#    json.dump(questions_generated, fout)

# COMMAND ----------

# to read
#with open('data/questions.json') as json_file:
#    questions = json.load(json_file)

# COMMAND ----------

def check_Q_examples(doc):
    
    print(f"Title: {doc['title']}")
    print(f"Category: {doc['category']}")
    print(f"# of questions: {doc['questions_number']}")
    print(f"Questions: {doc['questions']}")
    print(f"Token usage: {doc['question_generation_token_usage']}")

# COMMAND ----------

check_Q_examples(questions_generated[0])

# COMMAND ----------

# MAGIC %md
# MAGIC Check actual token usage and compare to approximate cost

# COMMAND ----------

total_input_tokens = sum([i['question_generation_token_usage']['prompt_tokens'] for i in questions_generated])
input_cost = 0.01 / 1000
total_output_tokens = sum([i['question_generation_token_usage']['completion_tokens'] for i in questions_generated])
output_cost = 0.03 / 1000

cost_actual = (total_input_tokens * input_cost + total_output_tokens * output_cost)

print(f'Approx cost in USD to generate questions: {cost_approximation}')
print(f'Actual cost in USD to generate questions: {cost_actual}')

# COMMAND ----------

# MAGIC %md
# MAGIC Seems like I've overestimated price, questions were generated for a total cost of ~4 USD

# COMMAND ----------

# MAGIC %md
# MAGIC #### Flatten question list so that they can be fed 1-by-1 to `gpt-3.5-turbo` for answer generation

# COMMAND ----------

for i in questions_generated:

  i['questions_split'] = i['questions'].split('\n')
  i['questions_split'] = [i.strip() for i in i['questions_split']] # remove trailing ws if any
  i['questions_split'] = [i for i in i['questions_split'] if i != ''] # remove empty strings after splitting
  i['questions_split'] = [i.lstrip('0123456789.- ') for i in i['questions_split']] # remove numbering from string beginning

# COMMAND ----------

all_questions_flattened = len(sum([i['questions_split'] for i in questions_generated], []))
print(f"Total # of questions generated for QA SFT dataset: {all_questions_flattened}")

# COMMAND ----------

#with open('data/questions_flattened.json', 'w') as fout:
#    json.dump(questions_generated, fout)

# COMMAND ----------

!ls -lh data/

# COMMAND ----------

# MAGIC %md
# MAGIC #### Put together prompt for Answer generation

# COMMAND ----------

def AnswerGenerator(context, category, title, question):

    system_message = """You are an expert Q&A system that is trusted around the world."""

    instuction = f"""
    Always answer the question using the provided context information, and not prior knowledge. Some rules to follow:
    1. Never directly reference the given context in your answer.
    2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
    3. Give detailed, thorough, long answers, mention all information that may be important or relevant to both the question and your response.

    Context information is below: 
    The document is a {category}, it's title is {title}.

    {context}

    Given the context information and not prior knowledge, answer the question below:
    {question}

    Answer: 
    """

    messages = [{'role' : 'system', 'content' : system_message},
                {'role' : 'user', 'content' : instuction}]

    response, usage = generate_response(messages, deployment_name = deployment_name_gpt_35_turbo)

    return response, usage

# COMMAND ----------

title = questions_generated[66]['title']
category = questions_generated[66]['category']
context = questions_generated[66]['content']
question = questions_generated[66]['questions_split'][6]

answer, usage = AnswerGenerator(context, category, title, question)

print(f'Question: {question}')
print(f'Answer: {answer}')

# COMMAND ----------

# MAGIC %md
# MAGIC Approximate cost for answer generation

# COMMAND ----------

avg_input_tokens = 1000
input_cost = 0.0015 / 1000
avg_output_tokens = 200
output_cost = 0.002 / 1000

cost_approximation_answer = all_questions_flattened * (avg_input_tokens * input_cost + avg_output_tokens * output_cost)
print(f'Approx cost in USD to generate answers: {cost_approximation_answer}')

# COMMAND ----------

# MAGIC %md
# MAGIC Run answer generation

# COMMAND ----------

questions_answered = []

for i in tqdm(questions_generated):
    
    title = i['title']
    category = i['category']
    context = i['content']
    questions = i['questions_split']

    answers = []
    usages = []
    for question in questions:
        answer, usage = AnswerGenerator(context, category, title, question)
        answers.append(answer)
        usages.append(usage)

    usages = dict(reduce(add, map(Counter, usages)))

    output = {**i, 
              'answers' : answers,
              'answer_generation_token_usage' : usages}
    questions_answered.append(output)

    # sleep 2 secs after each API call
    time.sleep(2)

# COMMAND ----------

print(len(questions_answered))

with open('data/questions_and_answers_flattened.json', 'w') as fout:
    json.dump(questions_answered, fout)

# COMMAND ----------

!ls -lh data/

# COMMAND ----------

# MAGIC %md
# MAGIC Check actual token usage for answer generation

# COMMAND ----------

total_input_tokens = sum([i['answer_generation_token_usage']['prompt_tokens'] for i in questions_answered])
input_cost = 0.0015 / 1000
total_output_tokens = sum([i['answer_generation_token_usage']['completion_tokens'] for i in questions_answered])
output_cost = 0.002 / 1000

cost_actual_answer = (total_input_tokens * input_cost + total_output_tokens * output_cost)

print(f'Approx cost in USD to generate answers: {cost_approximation_answer}')
print(f'Actual cost in USD to generate answers: {cost_actual_answer}')

# COMMAND ----------

# MAGIC %md
# MAGIC Total cost to generate QA dataset: ~7.5$

# COMMAND ----------

print(f'Total cost of generating QA dataset: {cost_actual_answer + cost_actual}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create SFT dataset
# MAGIC
# MAGIC Have not selected model yet, but structure will be similar `{'instruction' : 'question', 'completion' : 'answer'}`
# MAGIC
# MAGIC Will save as `pandas.DataFrame`

# COMMAND ----------

df = pd.DataFrame(questions_answered)
df = df.explode(['questions_split', 'answers'])
df.to_csv('data/QA_dataset.csv', index = False, escapechar='\\')

df[['questions_split', 'answers']].head(2)

# COMMAND ----------

!ls -lh data/

# COMMAND ----------

# MAGIC %md
# MAGIC Check word count so that neither Q nor A will be too long for whatever OSS model I'll be fine-tuning

# COMMAND ----------

df['questions_split'].apply(lambda x: len(x.split())).hist(bins = 50)

# COMMAND ----------

df['answers'].apply(lambda x: len(x.split())).hist(bins = 100)

# COMMAND ----------


