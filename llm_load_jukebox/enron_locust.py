import os
import queue
import threading

import fastparquet
from config import CACHE_DIR, KAGGLE_DATASET, OUTPUT_FILE, PARQUET_DATASET_PATH
from dotenv import load_dotenv
from enron_preprocessor import download_enron_dataset
from interfaces import OllamaAPI, OpenAIAPI, measure_metrics
from locust import HttpUser, between, events, task
from questions import generate_question

MAX_QUEUE_SIZE = 1000  # Bounded queue to avoid excess buffering


import pandas as pd


def producer_thread(parquet_path, shared_queue, stop_event):
    """
    Producer thread: Loads the Parquet file chunk by chunk (row-group by row-group)
    and puts each row (representing an email) into a shared queue.
    Stops reading if stop_event is set.
    """
    print("[Producer] Initializing ParquetFile...")
    pf = fastparquet.ParquetFile(parquet_path)
    total_rows_enqueued = 0

    for row_group_index, chunk_df in enumerate(pf.iter_row_groups()):
        if stop_event.is_set():
            print(f"[Producer] stop_event set. Ending before row group {row_group_index}.")
            break

        print(f"[Producer] Row group loaded. "
              f"Number of rows in chunk: {len(chunk_df)}")

        for row_index, row in chunk_df.iterrows():
            if stop_event.is_set():
                print(f"[Producer] stop_event set during row group {row_group_index}, row {row_index}. Stopping.")
                break

            shared_queue.put(row)
            total_rows_enqueued += 1

        if stop_event.is_set():
            break

        print(f"[Producer] Finished row group {row_group_index}. "
              f"Total rows enqueued so far: {total_rows_enqueued}")

    print(f"[Producer] Completed. Total rows enqueued: {total_rows_enqueued}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called once when the test starts.
    Initializes the queue and starts the producer thread.
    """
    print("[TestStart] Beginning test setup...")

    load_dotenv()

    # Configurable API selection
    environment.api_key = os.getenv("OPENAI_API_KEY")
    environment.model = os.getenv("MODEL", "llama3.2")
    environment.host = os.getenv("HOST", "http://localhost:11434")

    # Configurable API selection
    environment.api_name = os.getenv("API_NAME", "ollama")
    if environment.api_name != "openai" and environment.api_name != "ollama":
        raise ValueError(f"Unknown API: {environment.api_name}")


    download_enron_dataset(CACHE_DIR, KAGGLE_DATASET, PARQUET_DATASET_PATH)

    # Bounded queue
    environment.shared_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    # Stop event for clean shutdown
    environment.stop_event = threading.Event()

    # Start producer in a separate thread
    environment.producer = threading.Thread(
        target=producer_thread,
        args=(PARQUET_DATASET_PATH, environment.shared_queue, environment.stop_event),
        daemon=True
    )
    environment.producer.start()
    print("[TestStart] Producer thread launched.")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called once when the test stops.
    Terminates the producer thread.
    """
    print("[TestStop] Stopping producer thread...")
    if hasattr(environment, "stop_event"):
        environment.stop_event.set()
    if hasattr(environment, "producer"):
        environment.producer.join()
    print("[TestStop] Producer thread stopped.")


class EnronUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between tasks (in seconds)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.environment.host:
            raise ValueError("No Hostname provided. Start Locust with the -host=http://your-host")

        # Select the API
        self.environment.api = None
        if self.environment.api_name == "openai":
            self.environment.api = OpenAIAPI(model=self.environment.model, host=self.environment.host, client=self.client)
        elif self.environment.api_name == "ollama":
            self.environment.api = OllamaAPI(model=self.environment.model, host=self.environment.host, client=self.client)
        else:
            raise ValueError(f"Unknown API: {self.environment.api_name}")

    @task
    def process_email_task(self):
        """
        Gets an email from the shared queue, generates a question and passes it to the LLM.
        """
        try:
            # Timeout=5 to avoid blocking indefinitely if the queue is empty
            email = self.environment.shared_queue.get(timeout=5)
        except queue.Empty:
            print("[Consumer] Queue is empty, no more emails to process.")
            return

        body = email.get("Body", "")
        sender = email.get("From", "")

        question = generate_question(body, sender)
        answer, metrics = measure_metrics(self.environment.api, body, question)
         

        events.request.fire(
            request_type="LLM",
            name="E2E Latency",
            response_time=metrics["time_to_last_token"],
            response_length=metrics["output_tokens"],
            exception=None
        )
        
        events.request.fire(
            request_type="LLM",
            name="TTFT",
            response_time=metrics["time_to_first_token"],
            response_length=metrics["input_tokens"],
            exception=None
        )

        subject = email.get("Subject", "No Subject")
        print(f"[Consumer] Processed subject='{subject}'. Q='{question}' -> A='{answer}'")

