import yaml
import pandas as pd
import random
import numpy as np
from typing import List, Dict, Union


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


def calculate_rating_counts(rating_distribution: Union[List[float], str], total_samples: int) -> Dict[int, int]:
    """
    Calculate the number of reviews for each rating based on the distribution

    Args:
        rating_distribution: Either a list of probabilities [p1, p2, p3, p4, p5] that sum to 1,
                           or a string like "normal", "uniform", "skewed_positive", "skewed_negative"
        total_samples: Total number of reviews to generate

    Returns:
        Dictionary mapping rating (1-5) to count of reviews
    """
    ratings = [1, 2, 3, 4, 5]

    # Handle list-based distribution
    if isinstance(rating_distribution, list):
        if len(rating_distribution) != 5:
            raise ValueError(
                "Rating distribution list must have exactly 5 elements for ratings 1-5")

        if not np.isclose(sum(rating_distribution), 1.0, atol=0.01):
            raise ValueError(
                f"Rating distribution must sum to 1.0, got {sum(rating_distribution)}")

        probabilities = rating_distribution

    # Handle named distributions
    elif isinstance(rating_distribution, str):
        dist_name = rating_distribution.lower()

        if dist_name == "uniform":
            # Equal probability for all ratings
            probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]

        elif dist_name == "normal":
            # Normal distribution centered around 3 (middle rating)
            probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]

        elif dist_name == "skewed_positive" or dist_name == "positive":
            # Skewed towards higher ratings (4-5 stars)
            probabilities = [0.05, 0.10, 0.15, 0.30, 0.40]

        elif dist_name == "skewed_negative" or dist_name == "negative":
            # Skewed towards lower ratings (1-2 stars)
            probabilities = [0.40, 0.30, 0.15, 0.10, 0.05]

        elif dist_name == "bimodal":
            # High ratings and low ratings, less middle
            probabilities = [0.25, 0.15, 0.10, 0.15, 0.35]
        else:
            raise ValueError(f"Unknown distribution name: {rating_distribution}. "
                             f"Supported: uniform, normal, skewed_positive, skewed_negative, bimodal")
    else:
        raise ValueError(
            "rating_distribution must be either a list of floats or a string")

    # Calculate counts for each rating
    counts = {}
    remaining = total_samples

    for i, (rating, prob) in enumerate(zip(ratings, probabilities)):
        if i == len(ratings) - 1:
            # Last rating gets remaining samples to ensure exact total
            counts[rating] = remaining
        else:
            count = round(prob * total_samples)
            counts[rating] = count
            remaining -= count

    return counts


def format_rating_distribution(rating_distribution: Union[List[float], str, Dict[int, int]], batch_size: int = 10) -> str:
    """
    Format rating distribution instructions for the prompt

    Args:
        rating_distribution: Either a list of probabilities, a distribution name, or a dict {rating: count}
        batch_size: Number of reviews to generate in this batch (ignored if dict is provided)

    Returns:
        Formatted string describing the rating distribution
    """
    # If rating_distribution is already a dict {rating: count}, use it directly
    if isinstance(rating_distribution, dict):
        counts = rating_distribution
        batch_size = sum(counts.values())
    else:
        counts = calculate_rating_counts(rating_distribution, batch_size)

    formatted_parts = []
    formatted_parts.append(
        f"Generate exactly {batch_size} reviews with the following rating distribution:")
    formatted_parts.append("")

    for rating in sorted(counts.keys()):
        count = counts[rating]
        if count > 0:
            formatted_parts.append(
                f"  - {count} review(s) with {rating} star(s)")

    formatted_parts.append("")
    formatted_parts.append(
        "IMPORTANT: You MUST follow this exact distribution of ratings in your output.")

    return "\n".join(formatted_parts)


def load_few_shot_examples(csv_path: str, num_examples: int = 5, rating_column: str = 'Rating', review_column: str = 'Review Text') -> List[Dict]:
    """
    Load random rows from the review dataset for few-shot examples

    Args:
        csv_path: Path to the CSV file containing reviews
        num_examples: Number of random examples to retrieve

    Returns:
        List of dictionaries containing review examples
    """
    df = pd.read_csv(csv_path, engine="python")
    df[rating_column] = df[rating_column].apply(normalize_rating)
    df = df[df[rating_column] != -1]
    grouped = df.groupby(rating_column, group_keys=False)
    sampled = grouped.apply(lambda x: x.sample(1)) if len(
        grouped) > 1 else df.sample(n=min(num_examples, len(df)))
    sample_df = sampled.sample(n=min(num_examples, len(sampled)))

    examples = []
    for _, row in sample_df.iterrows():
        # Extract rating number from "Rated X out of 5 stars" format
        rating = row[rating_column]

        examples.append({
            'rating': rating,
            'review_text': row[review_column]
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
    rating_distribution: Union[List[float], str, Dict[int, int]],
    few_shot_examples: List[Dict],
    batch_size: int = 10,
    prompt_template: str = None
) -> str:
    """
    Build the complete system prompt by injecting persona, characteristics, and examples

    Args:
        persona: Dictionary containing persona information (name, age, gender, occupation)
        review_characteristics: List of characteristics describing the desired review
        rating_distribution: Either list of probabilities, distribution name, or dict {rating: count}
        few_shot_examples: List of example reviews
        batch_size: Number of reviews to generate in this batch
        prompt_template: Optional custom prompt template (loads from file if not provided)

    Returns:
        Complete system prompt ready to use
    """
    if prompt_template is None:
        prompt_template = load_prompt_template()

    # Format review characteristics as bullet points
    characteristics_text = "\n".join(
        [f"- {char}" for char in review_characteristics])

    # Format rating distribution
    rating_dist_text = format_rating_distribution(
        rating_distribution, batch_size)
    # print(rating_dist_text)
    # Format few-shot examples
    examples_text = format_few_shot_examples(few_shot_examples)

    # Inject all values into the template
    system_prompt = prompt_template.format(
        name=persona.get('name', 'Anonymous'),
        age=persona.get('age', 'Unknown'),
        gender=persona.get('gender', 'Unknown'),
        occupation=persona.get('occupation', 'Unknown'),
        review_characteristics=characteristics_text,
        rating_distribution=rating_dist_text,
        few_shot_examples=examples_text,
        batch_size=batch_size
    )

    return system_prompt


CONFIG = load_config()
