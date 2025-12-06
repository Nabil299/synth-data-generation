import os
from dotenv import load_dotenv
from openai import OpenAI
from utils import CONFIG

class OpenAIClient:
    def __init__(self):
        """Initialize OpenAI client with configuration from config.yaml"""
        load_dotenv()
        model_config = CONFIG['model_configuration']
        self.base_url = model_config['base_url']
        self.model_name = model_config['model_name']
        self.api_key = os.getenv('API_KEY')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, system_prompt: str, user_prompt: str, top_p: float = 0.9, temperature: float = 0.7) -> str:
        """
        Generate a response using OpenAI API

        Args:
            system_prompt: The system prompt to set the context/behavior
            user_prompt: The user's prompt/query

        Returns:
            The generated response from the model
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")


openai_client = OpenAIClient()
