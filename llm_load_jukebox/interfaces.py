import json
import time
from abc import ABC, abstractmethod
from typing import Generator, Tuple

import openai
import pandas as pd
import requests
import tiktoken
from transformers import AutoTokenizer


class LLMInterface(ABC):
    @abstractmethod
    def stream_request(self, email_content: str, question: str) -> Generator[Tuple[str, float, float, int], None, None]:
        """
        Streams the response to a request and measures the timing for tokens.
        :param email_content: The email content
        :param question: The question for the email
        :yield: A token with the corresponding timestamp
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.
        :param text: The text to tokenize
        :return: The token count
        """
        pass

import os
import time
from typing import Generator, Tuple

import openai
import tiktoken


class CustomClientWrapper:
    """
    A custom client wrapper to integrate with OpenAI's request logic.
    This wrapper allows for using a custom HTTP client (e.g., Locust client).
    """

    def __init__(self, client):
        """
        Initialize the wrapper with a custom client.
        :param client: Custom HTTP client (e.g., Locust client).
        """
        self.client = client

    def request(self, method: str, url: str, headers: dict, data=None, json=None, stream=False):
        """
        Custom request method to use the provided client for HTTP requests.
        """
        if method.lower() == "post":
            return self.client.post(url, headers=headers, json=json, stream=stream)
        else:
            raise NotImplementedError(f"HTTP method {method} not supported by CustomClientWrapper.")


class OpenAIAPI:
    def __init__(self, model: str = "gpt-4o-mini", host: str = "https://api.openai.com/v1", client=None):
        """
        Initialize the OpenAIAPI class with a custom client and configuration.
        :param model: The OpenAI model to use (default: "gpt-4o-mini").
        :param endpoint_url: The OpenAI API endpoint URL (default: "https://api.openai.com/v1").
        :param client: Custom HTTP client (e.g., Locust client).
        """
    
        self.model = model
        self.host = host
        self.tokenizer = tiktoken.encoding_for_model(model)

        # Configure OpenAI with the custom client wrapper if provided
        if client:
            custom_wrapper = CustomClientWrapper(client)
            openai.request = custom_wrapper.request

    def stream_request(self, email_content: str, question: str) -> Generator[Tuple[str, float, float, int], None, None]:
        """
        Streams responses from the OpenAI API in real-time, using the custom client if provided.
        """
        prompt = f"""
        Here is the content of an email:

        {email_content}

        Question: {question}

        Please answer the question based on the content of the email.
        """
        try:
            input_tokens = self.get_token_count(prompt)
            start_time = time.perf_counter()
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                api_base=self.endpoint_url
            )
            for chunk in response:
                delta = time.perf_counter() - start_time
                yield chunk["choices"][0]["delta"].get("content", ""), delta, input_tokens
        except Exception as e:
            print(f"Error in OpenAI streaming request: {e}")

    def get_token_count(self, text: str) -> int:
        """
        Uses the tiktoken library to calculate the token count for OpenAI's GPT models.
        """
        return len(self.tokenizer.encode(text))


class OllamaAPI(LLMInterface):

    def __init__(self, model: str, host : str = "http://localhost:11434", client = None):
        self.model = model

        # Count number of tokens in prompt, assuming a llama model
        # FIXME make the count dependent on the model parameter
        # We are not using the meta llama tokenizer here, because it needs a login to huggingface
        # We've got a tiktoken conversion error, thats why we teill Transformers to load the Python/slow version of the tokenizer,
        # bypassing the fast “C++/Rust-based” pipeline that triggers the conversion error.
        # FIXME use fast version
        self.tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", use_fast=False)
        self.client = client
        self.host = host

    def stream_request(self, email_content: str, question: str) -> Generator[Tuple[str, float, float, int], None, None]:
        """
        Streams responses from the Ollama API in real-time.
        """
        prompt = (
            f"Here is the content of an email:\n\n{email_content}\n\n"
            f"Question: {question}\n\n"
            "Please answer the question based on the content of the email."
        )

        input_tokens = self.get_token_count(prompt)
        
        try:
            if self.client is None:
                http_client = requests
            else:
                http_client = self.client    

            start_time = time.perf_counter()    
            
            response = http_client.post(
                f"{self.host}/api/generate",
                json={
                   "model": self.model,
                    "prompt": prompt
                },
                headers={"Content-Type": "application/json"},
                stream=True
        )

            response.raise_for_status()
            for chunk in response.iter_lines(decode_unicode=True):
                    delta = time.perf_counter() - start_time
                    chunk_data = json.loads(chunk)  # Parse the JSON chunk
                    content = chunk_data.get("response", "")                  
                    yield content, start_time, delta, input_tokens

        except requests.exceptions.RequestException as e:
            return f"Error with the Ollama request: {e}"    

    def get_token_count(self, text: str) -> int:
        """
        Get the Token Count using Huggingface AutoTokenizer
        """
        encoded = self.tokenizer(text)
        return len(encoded["input_ids"])

import time


def measure_metrics(api: LLMInterface, email_content: str, question: str):
    """
    Measures various metrics for API performance:
    - Time-to-First-Token (TTFT)
    - Time-to-Last-Token (TTLT)
    - Output token count
    - Input token count
    """
    token_count = 0
    time_to_first_token = None
    time_to_last_token = None
    in_tokens = None

    # Stream the response
    response = ""
    for token, start_time, delta_from_start_time, input_tokens in api.stream_request(email_content, question):
        if not in_tokens:
            in_tokens = input_tokens
            time_to_first_token = delta_from_start_time
        response += token

    time_to_last_token = delta_from_start_time

    token_count = api.get_token_count(response)

    return response, {
        "input_tokens" : in_tokens,
        "output_tokens": token_count,
        "time_to_first_token": time_to_first_token,
        "time_to_last_token": time_to_last_token
    }



