# Databricks notebook source
# MAGIC %md
# MAGIC ### Experiment with finetuning an OSS LLM on Hifly data
# MAGIC
# MAGIC 1. Take selected folders / files from gdrive, upload to AZ Blob (manual)
# MAGIC 2. Load and chunk files (larger, gpt-3.5-turbo / gpt-4-turbo), store docs in Workspace (DBX nb)
# MAGIC 3. Generate QA pairs for SFT, store train dataset in Workspace (DBX nb)
# MAGIC 4. SFT selected OSS with train data (DBX nb; oss: mistral / zephyr and as such models)
# MAGIC 5. Use finetuned model to answer questions about custom dataset (limited to selected folders)
# MAGIC
# MAGIC Selected folders:
# MAGIC - G:\.shortcut-targets-by-id\0B23Ot7AW9q8dTHJCSU1MT1A2SzQ\5_Common\CV\_belepeskor
# MAGIC - G:\.shortcut-targets-by-id\1RNDv5RgyMoV7C4mg23EPoDF10wg5_mCS\7_Knowledge\Alapinfok\EN
# MAGIC - G:\.shortcut-targets-by-id\1H-lHHzp1rv3kGYKoaGXexCBqARJque3M\4_Teams\Advanced_Analytics_Team\Sales\Bemutatkozó anyagok\Hiflylabs_AA_Sales_Deck_20230124_MASTER
# MAGIC
# MAGIC This means this poc model will be able to answer questions about (most) colleagues CV's, basic company info (wifi password, 12 points) and the references of the advanced analytics team
