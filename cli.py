from utils import CONFIG, build_system_prompt, load_few_shot_examples
from client import openai_client
from models import  parse_review_batch


def configure_prompt(num_examples: int = 5, batch_size: int = 10):
    """
    Configure the system prompt with persona, characteristics, and few-shot examples

    Args:
        num_examples: Number of few-shot examples to retrieve from the dataset
        batch_size: Number of reviews to generate in this batch

    Returns:
        Complete system prompt ready to use
    """
    data_generation_configuration = CONFIG['data_generation_configuration']
    persona = data_generation_configuration['persona']
    review_characteristics = data_generation_configuration['review_characteristics']
    rating_distribution = data_generation_configuration['rating_distribution']

    # Load few-shot examples from the dataset
    few_shot_examples = load_few_shot_examples(
        csv_path='./dataset/Amazon_Reviews.csv',
        num_examples=num_examples
    )

    # Build the complete system prompt
    system_prompt = build_system_prompt(
        persona=persona,
        review_characteristics=review_characteristics,
        rating_distribution=rating_distribution,
        few_shot_examples=few_shot_examples,
        batch_size=batch_size
    )

    return system_prompt


def generate_review(user_prompt: str, num_examples: int = 5, batch_size: int = 10) -> str:
    """
    Generate a batch of reviews using the OpenAI client

    Args:
        user_prompt: The user's prompt describing what kind of review to generate
        num_examples: Number of few-shot examples to include
        batch_size: Number of reviews to generate in this batch

    Returns:
        Generated reviews from the model (JSON string)
    """
    system_prompt = configure_prompt(
        num_examples=num_examples, batch_size=batch_size)

    response = openai_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    return response


def generate_all_reviews(user_prompt: str, num_examples: int = 5):
    """
    Generate all reviews needed based on min_generated_samples from config

    Args:
        user_prompt: The user's prompt describing what kind of review to generate
        num_examples: Number of few-shot examples to include

    Returns:
        List of all generated reviews
    """
    data_generation_config = CONFIG['data_generation_configuration']
    min_generated_samples = data_generation_config['min_generated_samples']
    batch_size = 10  # Fixed batch size as specified

    num_batches = (min_generated_samples + batch_size -
                   1) // batch_size

    print(
        f"Generating {min_generated_samples} reviews in {num_batches} batches of {batch_size}...")
    print("=" * 60)

    all_reviews = []

    for batch_num in range(num_batches):
        # Calculate actual batch size for the last batch
        remaining = min_generated_samples - (batch_num * batch_size)
        current_batch_size = min(batch_size, remaining)

        print(
            f"\nBatch {batch_num + 1}/{num_batches} - Generating {current_batch_size} reviews...")

        try:
            response = generate_review(
                user_prompt=user_prompt,
                num_examples=num_examples,
                batch_size=current_batch_size
            )

            # Parse the JSON response
            import json
            batch_reviews = json.loads(response)
            all_reviews.append(batch_reviews)

            print(f"✓ Batch {batch_num + 1} completed successfully")

        except Exception as e:
            print(f"✗ Error in batch {batch_num + 1}: {str(e)}")
            continue

    print("\n" + "=" * 60)
    print(f"Generation complete! Total batches: {len(all_reviews)}")

    return all_reviews


if __name__ == "__main__":
    # Example usage
    user_input = "Generate reviews for Amazon products"
    result = generate_review(user_input, batch_size=10)
    print("\nRaw JSON response:")
    print(result)

    # Parse and validate using Pydantic
    try:
        batch = parse_review_batch(result)
        print(
            f"\n✓ Successfully generated and validated {len(batch.reviews)} reviews")
        print(f"Review IDs: {list(batch.reviews.keys())}")

        # Verify rating distribution
        from collections import Counter
        ratings = [batch.reviews[key].rating for key in batch.reviews]
        rating_counts = dict(Counter(ratings))
        print(f"Rating distribution: {rating_counts}")

        # Show first review as example
        first_review = batch.reviews['1']
        print(f"\nFirst review:")
        print(f"  Rating: {first_review.rating} stars")
        print(f"  Text: {first_review.review_text[:100]}...")

    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        import traceback
        traceback.print_exc()