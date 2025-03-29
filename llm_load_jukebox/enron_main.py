import csv
import os

from config import CACHE_DIR, KAGGLE_DATASET, OUTPUT_FILE, PARQUET_DATASET_PATH
from dotenv import load_dotenv
from email_processor import EmailProcessor
from enron_preprocessor import download_enron_dataset
from interfaces import LLMInterface, OllamaAPI
from tqdm import tqdm

load_dotenv()


def process_emails(api: LLMInterface):
    """
    Processes emails from the preprocessed dataset and collects performance metrics.
    """
    # Ensure the dataset exists
    if not os.path.exists(PARQUET_DATASET_PATH):
        print(f"Dataset not found at '{PARQUET_DATASET_PATH}'. Please run the preprocessing step first.")
        return

    # Initialize the EmailProcessor
    processor = EmailProcessor(api, CACHE_DIR)

    # Process the dataset and save results
    processor.process_batch(PARQUET_DATASET_PATH, OUTPUT_FILE, show_progress=True)


if __name__ == "__main__":
    # Step 1: Download and preprocess the Enron dataset
    download_enron_dataset(CACHE_DIR, KAGGLE_DATASET, PARQUET_DATASET_PATH)

    # Step 2: Instantiate the API (e.g., OllamaAPI or OpenAIAPI)
    api = OllamaAPI(model="llama3.2", host="http://localhost:11434")

    # Step 3: Process the emails and collect metrics
    process_emails(api)