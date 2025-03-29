import os

# Kaggle dataset information
KAGGLE_DATASET = "wcukierski/enron-email-dataset"

# Cache directory for storing dataset and results
CACHE_DIR = os.path.expanduser("~/.ml-mesh/enron-dataset")

# Path to the preprocessed Parquet dataset
PARQUET_DATASET_PATH = os.path.join(CACHE_DIR, "emails.parquet")

# Path to the output file for processed results
OUTPUT_FILE = os.path.join(CACHE_DIR, "enron_questions_answers.csv")