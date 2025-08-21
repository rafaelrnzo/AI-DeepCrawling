import os
import json
from typing import List, Dict, Tuple, Union
from pydantic import BaseModel, Field, ValidationError
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

import openai
from api_management import get_api_key
from assets import PROMPT_PAGINATION, PRICING, LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME

load_dotenv()
import logging

class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list, description="List of pagination URLs, including 'Next' button URL if present")

def calculate_pagination_price(token_counts: Dict[str, int], model: str) -> float:
    input_tokens = token_counts['input_tokens']
    output_tokens = token_counts['output_tokens']
    
    input_price = input_tokens * PRICING[model]['input']
    output_price = output_tokens * PRICING[model]['output']
    
    return input_price + output_price

def detect_pagination_elements(url: str, indications: str, selected_model: str, markdown_content: str) -> Tuple[Union[PaginationData, Dict, str], Dict, float]:
    try:
        prompt_pagination = PROMPT_PAGINATION+"\n The url of the page to extract pagination from   "+url+"if the urls that you find are not complete combine them intelligently in a way that fit the pattern **ALWAYS GIVE A FULL URL**"
        if indications != "":
            prompt_pagination +=PROMPT_PAGINATION+"\n\n these are the users indications that, pay special attention to them: "+indications+"\n\n below are the markdowns of the website: \n\n"
        else:
            prompt_pagination +=PROMPT_PAGINATION+"\n There are no user indications in this case just apply the logic described. \n\n below are the markdowns of the website: \n\n"

        if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            client = OpenAI(api_key=get_api_key('OPENAI_API_KEY'))
            completion = client.beta.chat.completions.parse(
                model=selected_model,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
                response_format=PaginationData
            )

            parsed_response = completion.choices[0].message.parsed

            encoder = tiktoken.encoding_for_model(selected_model)
            input_token_count = len(encoder.encode(markdown_content))
            output_token_count = len(encoder.encode(json.dumps(parsed_response.dict())))
            token_counts = {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count
            }

            pagination_price = calculate_pagination_price(token_counts, selected_model)

            return parsed_response, token_counts, pagination_price

        elif selected_model == "gemini-1.5-flash":
            genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": PaginationData
                }
            )
            prompt = f"{prompt_pagination}\n{markdown_content}"
            input_tokens = model.count_tokens(prompt)
            completion = model.generate_content(prompt)
            usage_metadata = completion.usage_metadata
            token_counts = {
                "input_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count
            }
            response_content = completion.text
            
            logging.info(f"Gemini Flash response type: {type(response_content)}")
            logging.info(f"Gemini Flash response content: {response_content}")
            
            try:
                parsed_data = json.loads(response_content)
                if isinstance(parsed_data, dict) and 'page_urls' in parsed_data:
                    pagination_data = PaginationData(**parsed_data)
                else:
                    pagination_data = PaginationData(page_urls=[])
            except json.JSONDecodeError:
                logging.error("Failed to parse Gemini Flash response as JSON")
                pagination_data = PaginationData(page_urls=[])

            pagination_price = calculate_pagination_price(token_counts, selected_model)

            return pagination_data, token_counts, pagination_price

        elif selected_model == "Llama3.1 8B":
            openai.api_key = "lm-studio"
            openai.api_base = "http://localhost:1234/v1"
            response = openai.ChatCompletion.create(
                model=LLAMA_MODEL_FULLNAME,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
                temperature=0.7,
            )
            response_content = response['choices'][0]['message']['content'].strip()
            try:
                pagination_data = json.loads(response_content)
            except json.JSONDecodeError:
                pagination_data = {"next_buttons": [], "page_urls": []}
            token_counts = {
                "input_tokens": response['usage']['prompt_tokens'],
                "output_tokens": response['usage']['completion_tokens']
            }
            pagination_price = calculate_pagination_price(token_counts, selected_model)

            return pagination_data, token_counts, pagination_price

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        logging.error(f"An error occurred in detect_pagination_elements: {e}")
        return PaginationData(page_urls=[]), {"input_tokens": 0, "output_tokens": 0}, 0.0





            
        