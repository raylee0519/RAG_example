import time
import openai
from tqdm import tqdm
import json
import pandas as pd
import os 
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_response_gpt(system_message, user_message, model_name, temperature, max_tokens, top_p, use_json_format=False, retry_attempts=3, delay=5):
    if use_json_format:
        json_format = {"type": "json_object"}

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    attempts = 0
    while attempts < retry_attempts:
        try:
            if use_json_format:
                response = openai.ChatCompletion.create(
                    model=model_name, 
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    response_format=json_format,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model_name, 
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )

            return response['choices'][0]['message']['content']
        except openai.error.InvalidRequestError as e:
            error_message = str(e)
            print(f"Invalid request error: {error_message}")
            return None
        except openai.error.APIError as e:
            print(f"API error on attempt {attempts+1}: {e}")
            attempts += 1
            time.sleep(delay)

    print("Max retries reached, skipping this data.")
    return None