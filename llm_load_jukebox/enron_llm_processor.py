import csv
import json
import os
import random
import re
import time
from email.parser import Parser

import openai
import pandas as pd
import requests
from dotenv import load_dotenv
from fastparquet import write
from kaggle.api.kaggle_api_extended import KaggleApi
from questions import generate_question
from tqdm import tqdm
from transformers import AutoTokenizer

# Load the OpenAI API key (if using a .env file)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Kaggle dataset information
KAGGLE_DATASET = "wcukierski/enron-email-dataset"
CACHE_DIR = os.path.expanduser("~/.ml-mesh/enron-dataset")
PARQUET_DATASET_PATH = os.path.join(CACHE_DIR, "emails.parquet")


# Parse an email in RFC-5322 format and return From, To, Subject, and Body
def parse_rfc5322_email(email_string):
    header_pattern = re.compile(r'^(From|To|Subject):\s*(.*)$', re.MULTILINE)
    header_body_split = re.compile(r'\n\n', re.MULTILINE)
    split_email = header_body_split.split(email_string, maxsplit=1)
    headers = split_email[0] if len(split_email) > 0 else ""
    body = split_email[1] if len(split_email) > 1 else ""
    
    parsed_headers = {"From": None, "To": None, "Subject": None}
    for match in header_pattern.finditer(headers):
        key = match.group(1).strip()
        value = match.group(2).strip()
        parsed_headers[key] = value
    
    return {
        "From": parsed_headers["From"],
        "To": parsed_headers["To"],
        "Subject": parsed_headers["Subject"],
        "Body": body.strip() if body.strip() else None
    }

def download_enron_dataset():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    dataset_zip = os.path.join(CACHE_DIR, "enron-dataset.zip")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    csv_file = os.path.join(CACHE_DIR, "emails.csv")

    # Check if the dataset already exists
    if not os.path.exists(PARQUET_DATASET_PATH):
        print(f"Downloading the Enron dataset and saving it to '{CACHE_DIR}'...")
        # Download with a progress bar
        api.dataset_download_files(KAGGLE_DATASET, path=CACHE_DIR, unzip=True)
        print("Download complete!")

        chunksize = 10000  # number of rows per chunk

        for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize)):
            # Apply the parsing function
            parsed_chunk = chunk.copy()
            parsed_chunk[['From', 'To', 'Subject', 'Body']] = parsed_chunk['message'].apply(parse_rfc5322_email).apply(pd.Series)

            # Drop the original 'message' column
            parsed_chunk = parsed_chunk.drop(columns=['message'])

            if i == 0:
                # Write the first chunk (create the file)
                write(PARQUET_DATASET_PATH, parsed_chunk, compression="SNAPPY")
            else:
                # Write subsequent chunks
                write(PARQUET_DATASET_PATH, parsed_chunk, compression="SNAPPY", append=True)
    else:
        print(f"The Enron dataset is already cached in '{CACHE_DIR}'.")



def ask_llm_ollama(email_content, question, model="llama3.2", url="http://localhost:11434/api/chat", client=None):
    """
    Sends a request to the Ollama API to answer a question based on an email, using streaming.

    Args:
        email_content (str): The content of the email.
        question (str): The question to ask.
        model (str): The Ollama model (default: "llama3.2").
        url (str): The URL of the Ollama server (default: "http://localhost:11434/api/chat").
        client: Locust client. If passed, it weill be used instead of the requests package 

    Returns:
        str: The full response content assembled from the stream.
    """
    prompt = (
        f"Here is the content of an email:\n\n{email_content}\n\n"
        f"Question: {question}\n\n"
        "Please answer the question based on the content of the email."
    )

    # Count number of tokens in prompt, assuming a llama model
    # FIXME make the count dependent on the model parameter
    # We are not using the meta llama tokenizer here, because it needs a login to huggingface
    # We've got a tiktoken conversion error, thats why we teill Transformers to load the Python/slow version of the tokenizer,
    # bypassing the fast “C++/Rust-based” pipeline that triggers the conversion error.
    # FIXME use fast version
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", use_fast=False)
    encoded = tokenizer(prompt)
    input_tokens = len(encoded["input_ids"])
    
    try:
        if client is None:
            http_client = requests
        else:
            http_client = client

        start_time = time.perf_counter()    
        
        response = http_client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            headers={"Content-Type": "application/json"},
            stream=True
        )

        response.raise_for_status()
        first_token_time = None

        # Process the response stream
        full_response = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:  # Ignore empty chunks
                try:
                    # Record the time we receive the first token/chunk
                    if first_token_time is None:
                       first_token_time = time.perf_counter()
                    chunk_data = json.loads(chunk)  # Parse the JSON chunk
                    content = chunk_data.get("message", {}).get("content", "")
                    full_response += content
                except json.JSONDecodeError:
                    print(f"Invalid chunk: {chunk}")

        end_time = time.perf_counter()

        # Get number of tokens in output, assuming a llama model
        # FIXME make the count dependent on the model parameter
        # We are not using the meta llama tokenizer here, because it needs a login to huggingface
        # We've got a tiktoken conversion error, thats why we teill Transformers to load the Python/slow version of the tokenizer,
        # bypassing the fast “C++/Rust-based” pipeline that triggers the conversion error.
        # FIXME use fast version
        tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", use_fast=False)
        encoded = tokenizer(full_response)
        output_tokens = len(encoded["input_ids"])

        ttft_secs = (first_token_time - start_time) if first_token_time else None
        e2e_latency_secs = end_time - start_time

        return full_response.strip(),{"End-to-End Latency" : (int)(e2e_latency_secs * 1000), "Time to First Token" : (int)(ttft_secs * 1000), "Output Tokens" : output_tokens, "Input Tokens" : input_tokens}

    except requests.exceptions.RequestException as e:
        return f"Error with the Ollama request: {e}"


# Function to interact with OpenAI LLM
def ask_llm_openai(email_content, question, model="gpt-4"):
    prompt = f"""
Here is the content of an email:

{email_content}

Question: {question}

Please answer the question based on the email content.
"""
    try:
        response = openai.ChatCompletion.create(
            model,
            messages=[
                {"role": "system", "content": "You are a helpful NLP model."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error with the OpenAI request: {e}")
        return None


# Main function
def process_emails():
    output_file = os.path.join(CACHE_DIR, "enron_questions_answers.csv")
    
    # Load the dataset
    try:
        emails = pd.read_parquet(PARQUET_DATASET_PATH, engine="pyarrow")
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return

    # Make sure the dataset contains the 'Body' column
    if 'Body' not in emails.columns:
        print("The dataset must contain a 'Body' column.")
        return

    # Initialize the CSV file with headers
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()

    # Process emails with a progress bar
    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        for _, email in tqdm(emails.iterrows(), total=len(emails), desc="Processing emails"):
            body = email.get("Body", "")
            sender = email.get("From", "")

            # Generate a random question about the email
            question = generate_question(body, sender)
            
            # Send the email body and question to the LLM
            answer = ask_llm_ollama(body, question)
            
            # Write results directly to the CSV file
            writer.writerow({
                "question": question,
                "answer": answer
            })

    print(f"Results saved to '{output_file}'.")


# Script execution
if __name__ == "__main__":
    # Download and cache the Enron dataset
    download_enron_dataset()

    # Process the emails
    process_emails()
