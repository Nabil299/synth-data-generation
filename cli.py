from utils import CONFIG, build_system_prompt, load_few_shot_examples, calculate_rating_counts
from client import openai_client
from models import parse_review_batch, ReviewBatch
from db.database import ReviewDatabase
from typing import Optional, Dict
from db.db_config import initialize_database
from guardrails import QualityGuardrails
from collections import defaultdict
from metrics import calculate_all_metrics, save_results, export_reviews_to_folder
import time

def configure_prompt(num_examples: int = 5, batch_size: int = 10, custom_rating_dist: Dict[int, int] = None):
    """
    Configure the system prompt with persona, characteristics, and few-shot examples

    Args:
        num_examples: Number of few-shot examples to retrieve from the dataset
        batch_size: Number of reviews to generate in this batch
        custom_rating_dist: Optional custom rating distribution as dict {rating: count}
                           If provided, overrides the default rating_distribution

    Returns:
        Complete system prompt ready to use
    """
    data_generation_configuration = CONFIG['data_generation_configuration']
    persona = data_generation_configuration['persona']
    review_characteristics = data_generation_configuration['review_characteristics']
    rating_distribution = data_generation_configuration['rating_distribution']
    csv_configuration = CONFIG['csv_configuration']
    csv_path = csv_configuration['csv_path']
    rating_column = csv_configuration['rating_column']
    review_column = csv_configuration['review_column']
    # Use custom distribution if provided, otherwise use default
    if custom_rating_dist is not None:
        # Convert custom distribution dict to list format for the prompt
        rating_distribution = custom_rating_dist

    # Load few-shot examples from the dataset
    few_shot_examples = load_few_shot_examples(
        csv_path=csv_path,
        num_examples=num_examples,
        rating_column=rating_column,
        review_column=review_column
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


def generate_review(user_prompt: str, model_name: str, num_examples: int = 5, batch_size: int = 10, custom_rating_dist: Dict[int, int] = None,use_schema:bool = True) -> str:
    """
    Generate a batch of reviews using the OpenAI client

    Args:
        user_prompt: The user's prompt describing what kind of review to generate
        model_name: Name of the model to use for generation
        num_examples: Number of few-shot examples to include
        batch_size: Number of reviews to generate in this batch
        custom_rating_dist: Optional custom rating distribution as dict {rating: count}

    Returns:
        Generated reviews from the model (JSON string)
    """
    system_prompt = configure_prompt(
        num_examples=num_examples,
        batch_size=batch_size,
        custom_rating_dist=custom_rating_dist
    )

    response = openai_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        response_schema=ReviewBatch.model_json_schema() if use_schema else None
    )

    return response


def _generate_and_validate_batch(
    user_prompt: str,
    num_examples: int,
    batch_size: int,
    guardrails: QualityGuardrails,
    db: ReviewDatabase,
    model_name: str,
    custom_rating_dist: Dict[int, int] = None,
    use_schema: bool = True
):
    """
    Generate a single batch of reviews and validate them

    Args:
        user_prompt: The user's prompt describing what kind of review to generate
        num_examples: Number of few-shot examples to include
        batch_size: Number of reviews to generate in this batch
        guardrails: QualityGuardrails instance for validation
        db: Database instance for insertion
        model_name: Name of the model generating reviews
        custom_rating_dist: Optional custom rating distribution as dict {rating: count}
        use_schema: Whether to use the schema for validation
    Returns:
        Tuple of (reviews_list, accepted, rejected, batch_stats, num_inserted)
    """
    response = generate_review(
        user_prompt=user_prompt,
        model_name=model_name,
        num_examples=num_examples,
        batch_size=batch_size,
        custom_rating_dist=custom_rating_dist,
        use_schema=use_schema
    )

    # Parse the JSON response
    batch_reviews = parse_review_batch(response)

    # Convert to list format
    reviews_list = []
    for review in batch_reviews.reviews:
        reviews_list.append({
            'rating': review.rating,
            'review_text': review.review_text
        })

    # Validate batch using guardrails
    accepted, rejected, batch_stats = guardrails.validate_batch(
        reviews_list)

    # Insert only accepted reviews with model_name
    num_inserted = 0
    if accepted:
        num_inserted = db.insert_batch(reviews=accepted, model_name=model_name)

    return reviews_list, accepted, rejected, batch_stats, num_inserted


def _run_initial_generation(
    user_prompt: str,
    min_generated_samples: int,
    batch_size: int,
    num_examples: int,
    db: ReviewDatabase,
    guardrails: QualityGuardrails,
    rating_distribution,
    model_name: str,
    use_schema: bool = True
):
    """
    Run the initial batch generation phase

    Args:
        user_prompt: The user's prompt
        min_generated_samples: Target number of samples
        batch_size: Size of each batch
        num_examples: Number of few-shot examples
        db: Database instance
        guardrails: QualityGuardrails instance
        rating_distribution: The configured rating distribution
        use_schema: Whether to use the schema for validation
    Returns:
        Tuple of (all_reviews, total_stats, rejected_by_rating)
    """
    num_batches = (min_generated_samples + batch_size - 1) // batch_size

    print(
        f"Generating {min_generated_samples} reviews in {num_batches} batches of {batch_size}...")
    print("=" * 70)

    all_reviews = []
    total_stats = {
        'total': 0,
        'accepted': 0,
        'duplicates': 0,
        'sentiment_mismatches': 0
    }

    # Track rejected reviews by rating
    rejected_by_rating = defaultdict(int)

    for batch_num in range(num_batches):
        # Calculate actual batch size for the last batch
        remaining = min_generated_samples - (batch_num * batch_size)
        current_batch_size = min(batch_size, remaining)

        print(f"\n--- Batch {batch_num + 1}/{num_batches} ---")
        print(f"Generating {current_batch_size} reviews...")

        # Calculate expected rating distribution for this batch
        expected_dist = calculate_rating_counts(
            rating_distribution, current_batch_size)

        try:
            reviews_list, accepted, rejected, batch_stats, num_inserted = _generate_and_validate_batch(
                user_prompt=user_prompt,
                num_examples=num_examples,
                batch_size=current_batch_size,
                guardrails=guardrails,
                db=db,
                model_name=model_name,
                custom_rating_dist=expected_dist,
                use_schema=use_schema
            )

            all_reviews.append(reviews_list)

            # Track rejected reviews by rating
            for rejected_review in rejected:
                rating = rejected_review['rating']
                rejected_by_rating[rating] += 1

            print(f"✓ Generated {len(reviews_list)} reviews")
            print(f"✓ Inserted {num_inserted} reviews to database")
            print(f"✓ Quality check complete:")
            print(f"  Accepted: {batch_stats['accepted']}")
            print(f"  Duplicates: {batch_stats['duplicates']}")
            print(
                f"  Sentiment mismatches: {batch_stats['sentiment_mismatches']}")

            # Update total stats
            total_stats['total'] += batch_stats['total']
            total_stats['accepted'] += batch_stats['accepted']
            total_stats['duplicates'] += batch_stats['duplicates']
            total_stats['sentiment_mismatches'] += batch_stats['sentiment_mismatches']

        except Exception as e:
            print(f"✗ Error in batch {batch_num + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return all_reviews, total_stats, rejected_by_rating


def _print_initial_stats(all_reviews: list, total_stats: dict, rejected_by_rating: Dict[int, int]):
    """Print statistics for the initial generation phase"""
    print("\n" + "=" * 70)
    print(f"Initial generation complete!")
    print(f"Total batches: {len(all_reviews)}")

    print(f"\nInitial Generation Statistics:")
    print(f"  Total generated: {total_stats['total']}")
    print(f"  Accepted (unique): {total_stats['accepted']}")
    print(f"  Duplicates removed: {total_stats['duplicates']}")
    print(f"  Sentiment mismatches: {total_stats['sentiment_mismatches']}")

    total_rejected = total_stats['duplicates'] + \
        total_stats['sentiment_mismatches']
    print(f"  Total rejected: {total_rejected}")

    acceptance_rate = (
        total_stats['accepted'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
    print(f"  Acceptance rate: {acceptance_rate:.1f}%")

    # Show rejected counts by rating
    if rejected_by_rating:
        print(f"\n  Rejected by rating:")
        for rating in sorted(rejected_by_rating.keys()):
            count = rejected_by_rating[rating]
            print(f"    {rating}-star: {count} rejected")


def _run_single_retry_batch(
    user_prompt: str,
    num_examples: int,
    batch_size: int,
    guardrails: QualityGuardrails,
    db: ReviewDatabase,
    batch_num: int,
    num_batches: int,
    model_name: str,
    custom_rating_dist: Dict[int, int] = None,
    use_schema: bool = True
):
    """
    Generate and validate a single retry batch

    Args:
        custom_rating_dist: Optional custom rating distribution to match rejected reviews
        model_name: Name of the model generating reviews
        use_schema: Whether to use the schema for validation
    Returns:
        Tuple of (batch_stats, rejected_reviews, success)
    """
    print(
        f"\n  Retry Batch {batch_num + 1}/{num_batches} (size: {batch_size})...")

    if custom_rating_dist:
        print(f"  Target rating distribution: {dict(custom_rating_dist)}")

    try:
        reviews_list, accepted, rejected, batch_stats, num_inserted = _generate_and_validate_batch(
            user_prompt=user_prompt,
            num_examples=num_examples,
            batch_size=batch_size,
            guardrails=guardrails,
            db=db,
            model_name=model_name,
            custom_rating_dist=custom_rating_dist,
            use_schema=use_schema
        )

        print(f"  ✓ Generated {len(reviews_list)} retry reviews")
        print(
            f"  ✓ Quality check: {batch_stats['accepted']} accepted, "
            f"{batch_stats['duplicates']} duplicates, "
            f"{batch_stats.get('sentiment_mismatches', 0)} sentiment mismatches"
        )

        return batch_stats, rejected, True

    except Exception as e:
        print(f"  ✗ Error in retry batch {batch_num + 1}: {str(e)}")
        return None, [], False


def _run_retry_mechanism(
    user_prompt: str,
    num_examples: int,
    min_generated_samples: int,
    max_retry_rounds: int,
    retry_batch_size: int,
    db: ReviewDatabase,
    guardrails: QualityGuardrails,
    rejected_by_rating: Dict[int, int],
    model_name: str,
    use_schema: bool = True
):
    """
    Run the retry mechanism to replace rejected reviews with rating-aware generation

    Args:
        user_prompt: The user's prompt
        num_examples: Number of few-shot examples
        min_generated_samples: Target number of samples
        max_retry_rounds: Maximum retry rounds
        retry_batch_size: Size of retry batches
        db: Database instance
        guardrails: QualityGuardrails instance
        rejected_by_rating: Dict mapping rating to count of rejected reviews
        use_schema: Whether to use the schema for validation
    Returns:
        dict: Retry statistics
    """
    total_rejected = sum(rejected_by_rating.values())

    if total_rejected <= 0 or max_retry_rounds <= 0:
        return {'total': 0, 'accepted': 0, 'duplicates': 0, 'sentiment_mismatches': 0}

    print(f"\n" + "=" * 70)
    print(
        f"RETRY MECHANISM: Attempting to replace {total_rejected} rejected reviews")
    print(f"Max retry rounds: {max_retry_rounds}")
    print("=" * 70)

    retry_stats = {
        'total': 0,
        'accepted': 0,
        'duplicates': 0,
        'sentiment_mismatches': 0
    }

    # Track what we still need by rating
    needed_by_rating = dict(rejected_by_rating)

    for retry_round in range(max_retry_rounds):
        # Calculate how many samples we still need overall
        current_unique = db.get_review_count(model_name=model_name)
        still_needed = min_generated_samples - current_unique

        if still_needed <= 0:
            print(f"\n✓ Target reached! No more retries needed.")
            break

        print(f"\n--- Retry Round {retry_round + 1}/{max_retry_rounds} ---")
        print(
            f"Current unique reviews: {current_unique}/{min_generated_samples}")
        print(f"Still needed: {still_needed}")

        # Show what we need by rating
        if needed_by_rating:
            print(f"Needed by rating:")
            for rating in sorted(needed_by_rating.keys()):
                count = needed_by_rating[rating]
                if count > 0:
                    print(f"  {rating}-star: {count}")

        # Calculate number of batches - process rating-specific batches
        total_needed = sum(needed_by_rating.values())
        num_retry_batches = (
            total_needed + retry_batch_size - 1) // retry_batch_size
        print(f"Generating {num_retry_batches} retry batches...")

        round_generated = 0
        round_accepted = 0
        round_duplicates = 0
        round_sentiment_mismatches = 0

        for retry_batch_num in range(num_retry_batches):
            # Check if we already have enough
            current_unique = db.get_review_count(model_name=model_name)
            if current_unique >= min_generated_samples:
                print(f"✓ Target reached during retry!")
                break

            # Build rating distribution for this retry batch
            # Take up to retry_batch_size reviews from needed_by_rating
            custom_rating_dist = {}
            batch_total = 0

            for rating in sorted(needed_by_rating.keys()):
                if needed_by_rating[rating] > 0 and batch_total < retry_batch_size:
                    take = min(needed_by_rating[rating],
                               retry_batch_size - batch_total)
                    custom_rating_dist[rating] = take
                    batch_total += take

            if batch_total == 0:
                # Nothing left to retry
                break

            batch_stats, rejected, success = _run_single_retry_batch(
                user_prompt=user_prompt,
                num_examples=num_examples,
                batch_size=batch_total,
                guardrails=guardrails,
                db=db,
                batch_num=retry_batch_num,
                num_batches=num_retry_batches,
                model_name=model_name,
                custom_rating_dist=custom_rating_dist,
                use_schema=use_schema
            )

            if success and batch_stats:
                round_generated += batch_stats['total']
                round_accepted += batch_stats['accepted']
                round_duplicates += batch_stats['duplicates']
                round_sentiment_mismatches += batch_stats.get(
                    'sentiment_mismatches', 0)

                # Update what we still need based on what was accepted
                # Decrement the ratings we successfully generated
                for rating in custom_rating_dist.keys():
                    if rating in needed_by_rating:
                        # We attempted to generate this many, reduce from needed
                        needed_by_rating[rating] = max(
                            0, needed_by_rating[rating] - custom_rating_dist[rating])

                # Add back the rejected ones to needed_by_rating
                for rejected_review in rejected:
                    rating = rejected_review['rating']
                    needed_by_rating[rating] = needed_by_rating.get(
                        rating, 0) + 1

        # Update overall retry stats
        retry_stats['total'] += round_generated
        retry_stats['accepted'] += round_accepted
        retry_stats['duplicates'] += round_duplicates
        retry_stats['sentiment_mismatches'] += round_sentiment_mismatches

        print(f"\n  Round {retry_round + 1} Summary:")
        print(f"    Generated: {round_generated}")
        print(f"    Accepted: {round_accepted}")
        print(f"    Duplicates: {round_duplicates}")
        print(f"    Sentiment mismatches: {round_sentiment_mismatches}")
        acceptance_rate = (round_accepted / round_generated *
                           100) if round_generated > 0 else 0
        print(f"    Acceptance rate: {acceptance_rate:.1f}%")

        # Check if we've reached the target
        current_unique = db.get_review_count()
        if current_unique >= min_generated_samples:
            print(
                f"\n✓ Target of {min_generated_samples} unique reviews reached!")
            break

    _print_retry_summary(retry_stats)
    return retry_stats


def _print_retry_summary(retry_stats: dict):
    """Print summary of retry mechanism"""
    print(f"\n" + "=" * 70)
    print(f"RETRY MECHANISM COMPLETE")
    print("=" * 70)
    print(f"Retry Statistics:")
    print(f"  Total retry attempts: {retry_stats['total']}")
    print(f"  New unique reviews: {retry_stats['accepted']}")
    print(f"  Retry duplicates: {retry_stats['duplicates']}")

    acceptance_rate = (
        retry_stats['accepted'] / retry_stats['total'] * 100) if retry_stats['total'] > 0 else 0
    print(f"  Retry acceptance rate: {acceptance_rate:.1f}%")


def _print_final_stats(
    db: ReviewDatabase,
    min_generated_samples: int,
    total_stats: dict,
    retry_stats: dict,
    model_name: str
):
    """Print final statistics and results"""
    final_count = db.get_review_count(model_name=model_name)

    print(f"\n" + "=" * 70)
    print(f"FINAL DATABASE STATISTICS ({model_name})")
    print("=" * 70)
    print(f"  Total unique reviews in database: {final_count}")
    print(f"  Target: {min_generated_samples}")

    total_generated = total_stats['total'] + retry_stats.get('total', 0)
    total_duplicates = total_stats['duplicates'] + \
        retry_stats.get('duplicates', 0)
    total_sentiment_mismatches = total_stats['sentiment_mismatches'] + \
        retry_stats.get('sentiment_mismatches', 0)
    total_accepted = total_stats['accepted'] + retry_stats.get('accepted', 0)

    print(f"  Total generated (all attempts): {total_generated}")
    print(f"  Total duplicates detected: {total_duplicates}")
    print(f"  Total sentiment mismatches: {total_sentiment_mismatches}")

    acceptance_rate = (total_accepted / total_generated *
                       100) if total_generated > 0 else 0
    print(f"  Overall acceptance rate: {acceptance_rate:.1f}%")

    # Check if target was met
    if final_count >= min_generated_samples:
        print(
            f"\n✅ SUCCESS: Target of {min_generated_samples} unique reviews achieved!")
    else:
        shortfall = min_generated_samples - final_count
        print(f"\n⚠️  SHORTFALL: {shortfall} reviews short of target")
        print(f"   Consider increasing max_retry_rounds in config.yaml")


def generate_all_reviews(user_prompt: str, model_name: str):
    """
    Generate all reviews needed based on min_generated_samples from config

    Args:
        user_prompt: The user's prompt describing what kind of review to generate
        model_name: Name of the model to use for generation

    Returns:
        List of all generated reviews
    """
    # Load configuration
    data_generation_config = CONFIG['data_generation_configuration']
    min_generated_samples = data_generation_config['min_generated_samples']
    batch_size = data_generation_config['batch_size']
    num_examples = data_generation_config['num_few_shot_examples']
    max_retry_rounds = data_generation_config.get('max_retry_rounds', 3)
    retry_batch_size = data_generation_config.get('retry_batch_size', 10)
    rating_distribution = data_generation_config['rating_distribution']
    use_schema = data_generation_config.get('use_schema', True)
    # Initialize database and quality guardrails (with model_name filtering for duplicates)
    db = initialize_database()
    guardrails = QualityGuardrails(db=db, config=CONFIG, model_name=model_name)

    print(f"\n🤖 Using model: {model_name}")

    # Run initial generation
    all_reviews, total_stats, rejected_by_rating = _run_initial_generation(
        user_prompt=user_prompt,
        min_generated_samples=min_generated_samples,
        batch_size=batch_size,
        num_examples=num_examples,
        db=db,
        guardrails=guardrails,
        rating_distribution=rating_distribution,
        model_name=model_name,
        use_schema=use_schema
    )

    # Print initial statistics
    _print_initial_stats(all_reviews, total_stats, rejected_by_rating)

    # Run retry mechanism if needed (rating-aware)
    retry_stats = _run_retry_mechanism(
        user_prompt=user_prompt,
        num_examples=num_examples,
        min_generated_samples=min_generated_samples,
        max_retry_rounds=max_retry_rounds,
        retry_batch_size=retry_batch_size,
        db=db,
        guardrails=guardrails,
        rejected_by_rating=rejected_by_rating,
        model_name=model_name,
        use_schema=use_schema
    )

    # Print final statistics
    _print_final_stats(db, min_generated_samples,
                       total_stats, retry_stats, model_name)

    return all_reviews


if __name__ == "__main__":


    # Get configuration
    data_generation_config = CONFIG['data_generation_configuration']
    user_input = data_generation_config['user_prompt']

    # Support multiple models
    model_config = CONFIG.get('model_configuration', {})
    model_names = model_config.get('model_name')

    # Convert to list if single model name provided
    if isinstance(model_names, str):
        model_names = [model_names]

    print("\n" + "=" * 70)
    print("SYNTHETIC REVIEW GENERATION PIPELINE")
    print("=" * 70)
    print(f"Models to process: {len(model_names)}")
    for i, name in enumerate(model_names, 1):
        print(f"  {i}. {name}")
    print("=" * 70)

    # Process each model
    for model_idx, model_name in enumerate(model_names, 1):
        print(f"\n{'#' * 70}")
        print(
            f"# PROCESSING MODEL {model_idx}/{len(model_names)}: {model_name}")
        print(f"{'#' * 70}\n")

        generation_start = time.time()

        # Generate reviews for this model
        all_reviews = generate_all_reviews(user_input, model_name)

        generation_time = time.time() - generation_start
        print(
            f"\n✓ Generation completed in {int(generation_time//60)}m {int(generation_time%60)}s")

        # Calculate and save metrics for this model
        print(f"\n{'='*70}")
        print(f"CALCULATING QUALITY METRICS FOR: {model_name}")
        print(f"{'='*70}")

        try:
            metrics_results = calculate_all_metrics(model_name=model_name,total_time=generation_time)
            output_dir, json_path, md_path = save_results(
                metrics_results, model_name)
            reviews_csv = export_reviews_to_folder(
                output_dir, model_name=model_name)

            print(f"\n✅ All outputs saved for {model_name}:")
            print(f"   📁 Directory: {output_dir}")
            print(f"   📊 Quality Report: QUALITY_REPORT.md")
            print(f"   📄 Reviews CSV: generated_reviews.csv")
            print(f"   📋 Metrics JSON: quality_metrics.json")
        except Exception as e:
            print(f"\n❌ Error calculating metrics for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#' * 70}")
    print(f"# ALL MODELS PROCESSED SUCCESSFULLY!")
    print(f"{'#' * 70}\n")
