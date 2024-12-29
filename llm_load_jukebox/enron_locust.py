import os
import queue
import threading

import fastparquet
from dotenv import load_dotenv
from enron_llm_processor import (
    PARQUET_DATASET_PATH,
    ask_llm_ollama,
    download_enron_dataset,
)
from locust import HttpUser, between, events, task
from questions import generate_question

MAX_QUEUE_SIZE = 1000  # Bounded queue to avoid excess buffering

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

    download_enron_dataset()

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
        answer = ask_llm_ollama(body, question, client=self.client)

        subject = email.get("Subject", "No Subject")
        print(f"[Consumer] Processed subject='{subject}'. Q='{question}' -> A='{answer}'")

