import os
import re

import pandas as pd
from fastparquet import write
from kaggle.api.kaggle_api_extended import KaggleApi


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

def download_enron_dataset(cache_dir, kaggle_dataset, parquet_path):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    dataset_zip = os.path.join(cache_dir, "enron-dataset.zip")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    csv_file = os.path.join(cache_dir, "emails.csv")

    # Check if the dataset already exists
    if not os.path.exists(parquet_path):
        print(f"Downloading the Enron dataset and saving it to '{cache_dir}'...")
        # Download with a progress bar
        api.dataset_download_files(kaggle_dataset, path=cache_dir, unzip=True)
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
                write(parquet_path, parsed_chunk, compression="SNAPPY")
            else:
                # Write subsequent chunks
                write(parquet_path, parsed_chunk, compression="SNAPPY", append=True)
    else:
        print(f"The Enron dataset is already cached in '{cache_dir}'.")