import yaml
import pandas as pd
import random
from typing import List, Dict


def load_config(config_path: str = './configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompt_template(prompt_path: str = './configs/prompt.yaml') -> str:
    with open(prompt_path, 'r') as f:
        prompt_data = yaml.safe_load(f)
        return prompt_data['system_prompt']

def normalize_rating(rating: str) -> int:
    if not rating:
        return -1
    return int(rating.split()[1]) if 'Rated' in rating else 3

def load_few_shot_examples(csv_path: str, num_examples: int = 5) -> List[Dict]:
    """
    Load random rows from the review dataset for few-shot examples

    Args:
        csv_path: Path to the CSV file containing reviews
        num_examples: Number of random examples to retrieve

    Returns:
        List of dictionaries containing review examples
    """
    df = pd.read_csv(csv_path,engine="python")
    df['Rating'] = df['Rating'].apply(normalize_rating)
    df = df[df['Rating'] != -1]
    grouped = df.groupby('Rating', group_keys=False)
    sampled = grouped.apply(lambda x: x.sample(1)) if len(grouped) > 1 else df.sample(n=min(num_examples, len(df)))
    sample_df = sampled.sample(n=min(num_examples, len(sampled)))

    examples = []
    for _, row in sample_df.iterrows():
        # Extract rating number from "Rated X out of 5 stars" format
        rating = row['Rating']

        examples.append({
            'rating': rating,
            'review_text': row['Review Text']
        })

    return examples


def format_few_shot_examples(examples: List[Dict]) -> str:
    """
    Format few-shot examples into a readable string for the prompt

    Args:
        examples: List of example dictionaries

    Returns:
        Formatted string with examples
    """
    formatted = []
    for i, example in enumerate(examples, 1):
        formatted.append(f"""Example {i}:
Rating: {example['rating']}
Review: {example['review_text']}
""")

    return "\n".join(formatted)


def build_system_prompt(
    persona: Dict[str, any],
    review_characteristics: List[str],
    few_shot_examples: List[Dict],
    prompt_template: str = None
) -> str:
    """
    Build the complete system prompt by injecting persona, characteristics, and examples

    Args:
        persona: Dictionary containing persona information (name, age, gender, occupation)
        review_characteristics: List of characteristics describing the desired review
        few_shot_examples: List of example reviews
        prompt_template: Optional custom prompt template (loads from file if not provided)

    Returns:
        Complete system prompt ready to use
    """
    if prompt_template is None:
        prompt_template = load_prompt_template()

    # Format review characteristics as bullet points
    characteristics_text = "\n".join(
        [f"- {char}" for char in review_characteristics])

    # Format few-shot examples
    examples_text = format_few_shot_examples(few_shot_examples)

    # Inject all values into the template
    system_prompt = prompt_template.format(
        name=persona.get('name', 'Anonymous'),
        age=persona.get('age', 'Unknown'),
        gender=persona.get('gender', 'Unknown'),
        occupation=persona.get('occupation', 'Unknown'),
        review_characteristics=characteristics_text,
        few_shot_examples=examples_text
    )

    return system_prompt


CONFIG = load_config()
