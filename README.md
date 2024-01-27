# Experiments with fine-tuning open-source LLMs

Base LLMs
  - mistral
  - zephyr

SFT approaches
  - PEFT (LoRA & qLoRA)

SFT datasets
  - custom built QA data from proprietary data

Additional infos
  - Infra: Databricks on Azure
  - Tracking: MLFlow
  - Adapters: HuggingFace (private repos)

Experiments
1. Educational demo - custom SFT dataset created from proprietary documents, then a Mistral-based LLM fine-tuned on it
2. Personal use case - LLM fine-tuned on facebook conversation history