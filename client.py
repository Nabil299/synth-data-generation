import os
from dotenv import load_dotenv
from openai import OpenAI
from utils import CONFIG
from pydantic import BaseModel


class OpenAIClient:
    def __init__(self):
        """Initialize OpenAI client with configuration from config.yaml"""
        load_dotenv()
        model_config = CONFIG['model_configuration']
        self.base_url = model_config['base_url']
        self.default_model_name = model_config['model_name']
        self.api_key = os.getenv('API_KEY')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, system_prompt: str, user_prompt: str, model_name: str = None, top_p: float = 0.9, temperature: float = 0.7, response_schema: BaseModel = None) -> str:
        """
        Generate a response using OpenAI API

        Args:
            system_prompt: The system prompt to set the context/behavior
            user_prompt: The user's prompt/query
            model_name: Optional model name to use (defaults to config value)
            top_p: Top-p sampling parameter
            temperature: Temperature parameter
            response_schema: Pydantic model for structured output

        Returns:
            The generated response from the model
        """
        # Use provided model_name or fall back to default
        model_to_use = model_name if model_name else self.default_model_name

        # If default is a list, use the first one
        if isinstance(model_to_use, list):
            model_to_use = model_to_use[0]

        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                response_format={
                    "type": "json_schema",
                    "json_schema": response_schema
                },
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(
                f"Error generating response with model {model_to_use}: {str(e)}")


openai_client = OpenAIClient()
