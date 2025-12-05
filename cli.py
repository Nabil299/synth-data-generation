from utils import CONFIG, build_system_prompt, load_few_shot_examples
from client import openai_client


def configure_prompt(num_examples: int = 5):
    """
    Configure the system prompt with persona, characteristics, and few-shot examples

    Args:
        num_examples: Number of few-shot examples to retrieve from the dataset

    Returns:
        Complete system prompt ready to use
    """
    data_generation_configuration = CONFIG['data_generation_configuration']
    persona = data_generation_configuration['persona']
    review_characteristics = data_generation_configuration['review_characteristics']

    # Load few-shot examples from the dataset
    few_shot_examples = load_few_shot_examples(
        csv_path='./dataset/Amazon_Reviews.csv',
        num_examples=num_examples
    )

    # Build the complete system prompt
    system_prompt = build_system_prompt(
        persona=persona,
        review_characteristics=review_characteristics,
        few_shot_examples=few_shot_examples
    )

    return system_prompt


def generate_review(user_prompt: str, num_examples: int = 5):
    """
    Generate a review using the OpenAI client

    Args:
        user_prompt: The user's prompt describing what kind of review to generate
        num_examples: Number of few-shot examples to include

    Returns:
        Generated review from the model
    """
    system_prompt = configure_prompt(num_examples=num_examples)
    print(system_prompt)
    # response = openai_client.generate(
    #     system_prompt=system_prompt,
    #     user_prompt=user_prompt
    # )
    # return response


if __name__ == "__main__":
    # Example usage
    user_input = "Generate a review for an Amazon product"
    result = generate_review(user_input)
    print(result)
