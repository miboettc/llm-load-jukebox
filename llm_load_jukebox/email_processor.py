import csv
import os

import pandas as pd
from interfaces import LLMInterface, measure_metrics
from questions import generate_question
from tqdm import tqdm


class EmailProcessor:
    """
    The EmailProcessor class is responsible for processing emails from a dataset
    and interacting with a language model (LLM) to generate responses and collect metrics.

    Key Responsibilities:
    - Process a single email and generate a response along with performance metrics.
    - Process a batch of emails from a dataset and save the results to a CSV file.
    - Interact with an LLM through the provided API interface to generate responses.

    Attributes:
    - api (LLMInterface): The API interface for interacting with the language model.
    - cache_dir (str): The directory where intermediate or output files are stored.

    Methods:
    - process_single_email(email): Processes a single email, generates a question, 
      interacts with the LLM, and returns the response and metrics.
    - process_batch(dataset_path, output_file, show_progress): Processes a batch of emails 
      from a dataset, generates responses and metrics, and saves the results to a CSV file.
    """
        
    def __init__(self, api: LLMInterface, cache_dir: str):
        self.api = api
        self.cache_dir = cache_dir
        
    def process_single_email(self, email):
        """Process a single email and return metrics"""
        body = email.get("Body", "")
        sender = email.get("From", "")
        email_id = email.get("Message-ID", "unknown")

        # Generate question
        question = generate_question(sender, body)
        
        try:
            answer, metrics = measure_metrics(self.api, body, question)
            return {
                "email_id": email_id,
                "question": question,
                "answer": answer,
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "time_to_first_token": metrics["time_to_first_token"],
                "time_to_last_token": metrics["time_to_last_token"]
            }
        except Exception as e:
            print(f"Error processing email ID {email_id}: {e}")
            return None

    def process_batch(self, dataset_path: str, output_file: str, show_progress: bool = True):
        """Process a batch of emails from a dataset"""
        try:
            emails = pd.read_parquet(dataset_path, engine="pyarrow")
        except Exception as e:
            print(f"Error loading the dataset: {e}")
            return False

        if 'Body' not in emails.columns:
            print("The dataset must contain a 'Body' column.")
            return False

        # Initialize output file
        fieldnames = ["email_id", "question", "answer", "input_tokens", "output_tokens", 
                     "time_to_first_token", "time_to_last_token"]
        
        if not os.path.exists(output_file):
            with open(output_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        # Process emails
        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Use iterator with or without progress bar
            iterator = tqdm(emails.iterrows(), total=len(emails), desc="Processing emails") if show_progress else emails.iterrows()
            
            for _, email in iterator:
                result = self.process_single_email(email)
                if result:
                    writer.writerow(result)

        print(f"Results saved to '{output_file}'.")
        return True