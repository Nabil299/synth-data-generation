"""
Quality Metrics for Synthetic Review Generation

This module calculates comprehensive quality metrics:
1. Semantic Diversity Score
2. Real-vs-Synthetic Similarity Score
3. Sentiment-Rating Alignment Score
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from scipy.stats import pearsonr
from typing import List, Dict, Tuple
import json
import warnings
import time
from datetime import datetime
from db.db_config import initialize_database
from utils import CONFIG

warnings.filterwarnings('ignore')


class SemanticDiversityMetric:
    """
    Measures how different synthetic reviews are from each other.

    Formula: D = 1 - avg(cosine_similarity(e_i, e_j)) for i ≠ j

    Interpretation:
    - 0.0 → extremely repetitive
    - 0.3 → moderate diversity
    - 0.6+ → very diverse (good)
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with sentence transformer model"""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ Embedding model loaded")

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts"""
        print(f"Computing embeddings for {len(texts)} reviews...")
        embeddings = self.model.encode(
            texts, show_progress_bar=False, batch_size=32)
        return embeddings

    def calculate_diversity_score(self, embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate semantic diversity score

        Returns:
            Tuple of (diversity_score, similarity_matrix)
        """
        N = len(embeddings)

        if N < 2:
            return 0.0, np.array([[1.0]])

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Calculate average similarity (excluding diagonal)
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]

        avg_similarity = np.mean(similarities)
        diversity_score = 1 - avg_similarity

        return diversity_score, similarity_matrix

    def analyze(self, reviews: List[str]) -> Dict:
        """
        Complete semantic diversity analysis

        Returns:
            Dictionary with diversity metrics
        """
        embeddings = self.compute_embeddings(reviews)
        diversity_score, similarity_matrix = self.calculate_diversity_score(
            embeddings)

        # Get upper triangle similarities for statistics
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]

        results = {
            'diversity_score': float(diversity_score),
            'avg_similarity': float(1 - diversity_score),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'std_similarity': float(np.std(similarities)),
            'embeddings': embeddings  # Keep for reuse
        }

        return results

    def interpret_score(self, score: float) -> str:
        """Interpret the diversity score"""
        if score < 0.2:
            return "EXTREMELY REPETITIVE"
        elif score < 0.3:
            return "LOW DIVERSITY"
        elif score < 0.5:
            return "MODERATE DIVERSITY"
        elif score < 0.6:
            return "GOOD DIVERSITY"
        else:
            return "EXCELLENT DIVERSITY"


class RealVsSyntheticMetric:
    """
    Compares synthetic dataset to real reviews using nearest-neighbor similarity.

    Formula: avg_similarity = avg(max_cosine(synthetic, real))

    Interpretation:
    - Low (0.2–0.4) → unrealistic
    - Moderate (0.4–0.6) → balanced
    - High (0.6–0.8) → very realistic
    """

    def __init__(self, semantic_metric: SemanticDiversityMetric):
        """Reuse the semantic metric's model for embeddings"""
        self.model = semantic_metric.model

    def compute_real_vs_synthetic_similarity(
        self,
        synthetic_embeddings: np.ndarray,
        real_embeddings: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute nearest-neighbor similarity between synthetic and real reviews

        Returns:
            Tuple of (avg_similarity, nearest_similarities)
        """
        # Compute similarity matrix: synthetic x real
        similarity_matrix = cosine_similarity(
            synthetic_embeddings, real_embeddings)

        # For each synthetic review, find the highest similarity with real reviews
        nearest_similarities = np.max(similarity_matrix, axis=1)

        # Average across all synthetic reviews
        avg_similarity = np.mean(nearest_similarities)

        return avg_similarity, nearest_similarities

    def analyze(
        self,
        synthetic_reviews: List[str],
        real_reviews: List[str],
        synthetic_embeddings: np.ndarray = None
    ) -> Dict:
        """
        Complete real-vs-synthetic analysis

        Returns:
            Dictionary with similarity metrics
        """
        # Compute embeddings
        if synthetic_embeddings is None:
            print("Computing synthetic embeddings...")
            synthetic_embeddings = self.model.encode(
                synthetic_reviews, show_progress_bar=False, batch_size=32
            )

        print(
            f"Computing real review embeddings ({len(real_reviews)} samples)...")
        real_embeddings = self.model.encode(
            real_reviews, show_progress_bar=False, batch_size=32
        )

        avg_similarity, nearest_similarities = self.compute_real_vs_synthetic_similarity(
            synthetic_embeddings, real_embeddings
        )

        results = {
            'avg_similarity': float(avg_similarity),
            'realism_score': float(1 - avg_similarity),
            'min_similarity': float(np.min(nearest_similarities)),
            'max_similarity': float(np.max(nearest_similarities)),
            'std_similarity': float(np.std(nearest_similarities))
        }

        return results

    def interpret_score(self, avg_similarity: float) -> str:
        """Interpret the real-vs-synthetic similarity"""
        if avg_similarity < 0.15:
            return "VERY UNREALISTIC - Too different from real reviews"
        elif avg_similarity < 0.35:
            return "SOMEWHAT UNREALISTIC - Limited resemblance"  
        elif avg_similarity < 0.65:
            return "BALANCED - Good mix of realism and diversity"
        elif avg_similarity < 0.85:
            return "VERY REALISTIC - Strong resemblance"
        else:
            return "TOO SIMILAR - Possible overfitting/copying"


class SentimentRatingAlignmentMetric:
    """
    Checks if sentiment matches rating levels using Pearson correlation.

    Formula: S = corr(rating, sentiment)

    Interpretation:
    - < 0.40 → weak alignment
    - 0.50–0.70 → acceptable
    - 0.75+ → excellent
    """

    def __init__(self, sentiment_model='cardiffnlp/twitter-roberta-base-sentiment'):
        """Initialize sentiment analysis model"""
        print(f"Loading sentiment model: {sentiment_model}...")
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model=sentiment_model,
            device=-1  # Use CPU
        )
        print("✓ Sentiment model loaded")

    def compute_sentiment_scores(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentiment scores for texts using cardiffnlp/twitter-roberta

        Returns:
            Array of sentiment scores (0-1, where 1 is positive)
        """
        print(f"Computing sentiment for {len(texts)} reviews...")
        sentiment_scores = []

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Truncate long texts to avoid model limits
            batch = [text[:512] for text in batch]

            results = self.sentiment_analyzer(batch)

            for result in results:
                # Convert cardiffnlp output to 0-1 scale
                # LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
                label = result['label']
                confidence = result['score']

                if label == 'LABEL_0':  # Negative
                    score = (1 - confidence) * 0.5  # Map to 0-0.5 range
                elif label == 'LABEL_1':  # Neutral
                    score = 0.5  # Map to middle
                elif label == 'LABEL_2':  # Positive
                    score = 0.5 + (confidence * 0.5)  # Map to 0.5-1.0 range
                else:
                    score = 0.5

                sentiment_scores.append(score)

            if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
                print(
                    f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} reviews")

        print("✓ Sentiment computation complete")
        return np.array(sentiment_scores)

    def compute_alignment_score(
        self,
        ratings: np.ndarray,
        sentiments: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Pearson correlation between ratings and sentiments

        Returns:
            Tuple of (correlation, p_value)
        """
        correlation, p_value = pearsonr(ratings, sentiments)
        return correlation, p_value

    def analyze(self, reviews: List[str], ratings: List[int]) -> Dict:
        """
        Complete sentiment-rating alignment analysis

        Returns:
            Dictionary with alignment metrics
        """
        sentiments = self.compute_sentiment_scores(reviews)
        ratings_array = np.array(ratings)

        correlation, p_value = self.compute_alignment_score(
            ratings_array, sentiments)

        # Compute per-rating statistics
        rating_sentiment_stats = {}
        for rating in sorted(set(ratings)):
            mask = ratings_array == rating
            rating_sentiments = sentiments[mask]
            rating_sentiment_stats[rating] = {
                'mean': float(np.mean(rating_sentiments)),
                'std': float(np.std(rating_sentiments)),
                'min': float(np.min(rating_sentiments)),
                'max': float(np.max(rating_sentiments)),
                'count': int(len(rating_sentiments))
            }

        results = {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'rating_sentiment_stats': rating_sentiment_stats
        }

        return results

    def interpret_correlation(self, correlation: float) -> str:
        """Interpret the correlation score"""
        if correlation < 0.40:
            return "WEAK ALIGNMENT"
        elif correlation < 0.50:
            return "LOW ALIGNMENT"
        elif correlation < 0.70:
            return "ACCEPTABLE ALIGNMENT"
        elif correlation < 0.75:
            return "GOOD ALIGNMENT"
        else:
            return "EXCELLENT ALIGNMENT"


def load_synthetic_reviews(model_name: str = None):
    """
    Load synthetic reviews from database

    Args:
        model_name: Optional model name to filter reviews
    """
    db = initialize_database()
    reviews = db.get_all_reviews(model_name=model_name)
    db.close()

    df = pd.DataFrame(reviews)
    if model_name:
        print(
            f"✓ Loaded {len(df)} synthetic reviews from database (model: {model_name})")
    else:
        print(f"✓ Loaded {len(df)} synthetic reviews from database")
    return df


def load_real_reviews(csv_path='./dataset/Amazon_Reviews.csv', num_samples=50):
    """Load real reviews from CSV file for comparison"""
    df = pd.read_csv(csv_path, engine="python")

    # Normalize rating
    def normalize_rating(rating):
        if not rating or pd.isna(rating):
            return None
        if 'Rated' in str(rating):
            return int(str(rating).split()[1])
        return 3

    df['Rating'] = df['Rating'].apply(normalize_rating)
    df = df[df['Rating'].notna()]

    # Sample reviews
    sampled = df.sample(n=min(num_samples, len(df)), random_state=42)

    result = pd.DataFrame({
        'rating': sampled['Rating'].values,
        'review_text': sampled['Review Text'].values
    })

    print(f"✓ Loaded {len(result)} real reviews from CSV")
    return result


def calculate_all_metrics(model_name: str = None,total_time: float = 0):
    """
    Calculate all quality metrics and save results

    Args:
        model_name: Optional model name to filter reviews and include in results

    Returns:
        Dictionary with all metrics and timing information
    """
    print("=" * 70)
    print("SYNTHETIC REVIEW QUALITY METRICS")
    if model_name:
        print(f"Model: {model_name}")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    print("Loading data...")
    synthetic_df = load_synthetic_reviews(model_name=model_name)
    real_df = load_real_reviews(num_samples=50)

    print(f"\nDataset Summary:")
    print(f"  Synthetic reviews: {len(synthetic_df)}")
    print(f"  Real reviews: {len(real_df)}")
    print(f"\nSynthetic rating distribution:")
    print(synthetic_df['rating'].value_counts().sort_index())
    print()

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name if model_name else 'unknown',
            'num_synthetic_reviews': len(synthetic_df),
            'num_real_reviews': len(real_df),
            'synthetic_rating_distribution': synthetic_df['rating'].value_counts().sort_index().to_dict()
        },
        'metrics': {}
    }

    # 1️⃣ Semantic Diversity Score
    print("\n" + "=" * 70)
    print("1️⃣ SEMANTIC DIVERSITY SCORE")
    print("=" * 70)

    semantic_metric = SemanticDiversityMetric(model_name='all-MiniLM-L6-v2')
    semantic_results = semantic_metric.analyze(
        synthetic_df['review_text'].tolist())

    print(f"Diversity Score: {semantic_results['diversity_score']:.4f}")
    print(f"Average Similarity: {semantic_results['avg_similarity']:.4f}")
    print(f"Min Similarity: {semantic_results['min_similarity']:.4f}")
    print(f"Max Similarity: {semantic_results['max_similarity']:.4f}")
    print(f"Std Similarity: {semantic_results['std_similarity']:.4f}")
    print(
        f"Interpretation: {semantic_metric.interpret_score(semantic_results['diversity_score'])}")

    # Store results (without embeddings for JSON serialization)
    results['metrics']['semantic_diversity'] = {
        k: v for k, v in semantic_results.items() if k != 'embeddings'
    }
    results['metrics']['semantic_diversity']['interpretation'] = semantic_metric.interpret_score(
        semantic_results['diversity_score']
    )

    # 2️⃣ Real-vs-Synthetic Similarity Score
    print("\n" + "=" * 70)
    print("2️⃣ REAL-VS-SYNTHETIC SIMILARITY SCORE")
    print("=" * 70)

    real_vs_synthetic_metric = RealVsSyntheticMetric(semantic_metric)
    real_vs_synthetic_results = real_vs_synthetic_metric.analyze(
        synthetic_df['review_text'].tolist(),
        real_df['review_text'].tolist(),
        synthetic_embeddings=semantic_results['embeddings']
    )

    print(
        f"Average Nearest-Neighbor Similarity: {real_vs_synthetic_results['avg_similarity']:.4f}")
    print(
        f"Realism Score (1 - avg_sim): {real_vs_synthetic_results['realism_score']:.4f}")
    print(f"Min Similarity: {real_vs_synthetic_results['min_similarity']:.4f}")
    print(f"Max Similarity: {real_vs_synthetic_results['max_similarity']:.4f}")
    print(f"Std Similarity: {real_vs_synthetic_results['std_similarity']:.4f}")
    print(
        f"Interpretation: {real_vs_synthetic_metric.interpret_score(real_vs_synthetic_results['avg_similarity'])}")

    results['metrics']['real_vs_synthetic'] = real_vs_synthetic_results
    results['metrics']['real_vs_synthetic']['interpretation'] = real_vs_synthetic_metric.interpret_score(
        real_vs_synthetic_results['avg_similarity']
    )

    # 3️⃣ Sentiment-Rating Alignment Score
    print("\n" + "=" * 70)
    print("3️⃣ SENTIMENT-RATING ALIGNMENT SCORE")
    print("=" * 70)

    sentiment_metric = SentimentRatingAlignmentMetric()
    sentiment_results = sentiment_metric.analyze(
        synthetic_df['review_text'].tolist(),
        synthetic_df['rating'].tolist()
    )

    print(f"Pearson Correlation: {sentiment_results['correlation']:.4f}")
    print(f"P-value: {sentiment_results['p_value']:.6f}")
    print(
        f"Interpretation: {sentiment_metric.interpret_correlation(sentiment_results['correlation'])}")
    print(f"\nPer-Rating Sentiment Statistics:")
    for rating in sorted(sentiment_results['rating_sentiment_stats'].keys()):
        stats = sentiment_results['rating_sentiment_stats'][rating]
        print(
            f"  Rating {rating}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['count']}")

    results['metrics']['sentiment_rating_alignment'] = sentiment_results
    results['metrics']['sentiment_rating_alignment']['interpretation'] = sentiment_metric.interpret_correlation(
        sentiment_results['correlation']
    )

    results['metadata']['total_time_seconds'] = float(total_time)
    results['metadata']['total_time_formatted'] = f"{int(total_time // 60)}m {int(total_time % 60)}s"

    print("\n" + "=" * 70)
    print(
        f"TOTAL EXECUTION TIME: {results['metadata']['total_time_formatted']}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


def save_results(results: Dict, model_name: str, output_base_dir: str = './outputs'):
    """
    Save results to a folder structure with model name

    Args:
        results: Dictionary with all metrics
        model_name: Name of the model used for generation
        output_base_dir: Base directory for outputs

    Returns:
        Tuple of (output_dir, json_path, md_path)
    """
    import os

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create output directory: outputs/{model_name}_{timestamp}/
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    output_dir = os.path.join(
        output_base_dir, f"{safe_model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n✓ Created output directory: {output_dir}")

    # Save full results as JSON
    json_path = os.path.join(output_dir, "quality_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON metrics: quality_metrics.json")

    # Create markdown report
    md_path = os.path.join(output_dir, "QUALITY_REPORT.md")
    with open(md_path, 'w') as f:
        f.write("# 📊 Synthetic Review Quality Report\n\n")
        f.write(f"**Generated**: {results['metadata']['timestamp']}\n\n")
        f.write(f"**Model**: {model_name}\n\n")
        f.write(
            f"**Execution Time**: {results['metadata']['total_time_formatted']}\n\n")
        f.write("---\n\n")

        # Dataset Summary
        f.write("## 📁 Dataset Summary\n\n")
        f.write(
            f"- **Total Synthetic Reviews**: {results['metadata']['num_synthetic_reviews']}\n")
        f.write(
            f"- **Real Reviews (comparison)**: {results['metadata']['num_real_reviews']}\n\n")

        f.write("### Rating Distribution\n\n")
        f.write("| Rating | Count |\n")
        f.write("|--------|-------|\n")
        for rating in sorted(results['metadata']['synthetic_rating_distribution'].keys()):
            count = results['metadata']['synthetic_rating_distribution'][rating]
            f.write(f"| {rating}⭐ | {count} |\n")
        f.write("\n---\n\n")

        # Metric 1: Semantic Diversity
        f.write("## 1️⃣ Semantic Diversity Score\n\n")
        sem_div = results['metrics']['semantic_diversity']
        f.write(
            "**Purpose**: Measures how different synthetic reviews are from each other.\n\n")
        f.write(f"### Results\n\n")
        f.write(f"- **Diversity Score**: `{sem_div['diversity_score']:.4f}`\n")
        f.write(
            f"- **Average Similarity**: `{sem_div['avg_similarity']:.4f}`\n")
        f.write(f"- **Min Similarity**: `{sem_div['min_similarity']:.4f}`\n")
        f.write(f"- **Max Similarity**: `{sem_div['max_similarity']:.4f}`\n")
        f.write(f"- **Std Similarity**: `{sem_div['std_similarity']:.4f}`\n\n")

        interpretation = sem_div['interpretation']
        if 'EXCELLENT' in interpretation:
            emoji = "✅"
        elif 'GOOD' in interpretation:
            emoji = "✅"
        elif 'MODERATE' in interpretation:
            emoji = "⚠️"
        else:
            emoji = "❌"

        f.write(f"**Interpretation**: {emoji} **{interpretation}**\n\n")
        f.write("**Scale**:\n")
        f.write("- `< 0.2`: Extremely repetitive\n")
        f.write("- `0.2-0.3`: Low diversity\n")
        f.write("- `0.3-0.5`: Moderate diversity\n")
        f.write("- `0.5-0.6`: Good diversity\n")
        f.write("- `> 0.6`: Excellent diversity ✨\n\n")
        f.write("---\n\n")

        # Metric 2: Real vs Synthetic
        f.write("## 2️⃣ Real-vs-Synthetic Similarity Score\n\n")
        real_vs = results['metrics']['real_vs_synthetic']
        f.write(
            "**Purpose**: Compares synthetic dataset to real reviews for realism.\n\n")
        f.write(f"### Results\n\n")
        f.write(
            f"- **Average Nearest-Neighbor Similarity**: `{real_vs['avg_similarity']:.4f}`\n")
        f.write(
            f"- **Realism Score (1 - avg_sim)**: `{real_vs['realism_score']:.4f}`\n")
        f.write(f"- **Min Similarity**: `{real_vs['min_similarity']:.4f}`\n")
        f.write(f"- **Max Similarity**: `{real_vs['max_similarity']:.4f}`\n")
        f.write(f"- **Std Similarity**: `{real_vs['std_similarity']:.4f}`\n\n")

        interpretation = real_vs['interpretation']
        if 'BALANCED' in interpretation or 'REALISTIC' in interpretation:
            emoji = "✅"
        elif 'TOO SIMILAR' in interpretation:
            emoji = "⚠️"
        else:
            emoji = "❌"

        f.write(f"**Interpretation**: {emoji} **{interpretation}**\n\n")
        f.write("**Scale**:\n")
        f.write("- `< 0.4`: Unrealistic\n")
        f.write("- `0.4-0.6`: Balanced (ideal) ✨\n")
        f.write("- `0.6-0.8`: Very realistic\n")
        f.write("- `> 0.8`: Too similar (possible copying)\n\n")
        f.write("---\n\n")

        # Metric 3: Sentiment-Rating Alignment
        f.write("## 3️⃣ Sentiment-Rating Alignment Score\n\n")
        sent_align = results['metrics']['sentiment_rating_alignment']
        f.write("**Purpose**: Validates if sentiment matches rating levels.\n\n")
        f.write(f"### Results\n\n")
        f.write(
            f"- **Pearson Correlation**: `{sent_align['correlation']:.4f}`\n")
        f.write(f"- **P-value**: `{sent_align['p_value']:.6f}`\n\n")

        interpretation = sent_align['interpretation']
        if 'EXCELLENT' in interpretation or 'GOOD' in interpretation:
            emoji = "✅"
        elif 'ACCEPTABLE' in interpretation:
            emoji = "⚠️"
        else:
            emoji = "❌"

        f.write(f"**Interpretation**: {emoji} **{interpretation}**\n\n")

        f.write("### Per-Rating Sentiment Statistics\n\n")
        f.write("| Rating | Mean | Std | Min | Max | Count |\n")
        f.write("|--------|------|-----|-----|-----|-------|\n")
        for rating in sorted(sent_align['rating_sentiment_stats'].keys()):
            stats = sent_align['rating_sentiment_stats'][rating]
            f.write(f"| {rating}⭐ | {stats['mean']:.3f} | {stats['std']:.3f} | "
                    f"{stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n")

        f.write("\n**Scale**:\n")
        f.write("- `< 0.40`: Weak alignment\n")
        f.write("- `0.50-0.70`: Acceptable alignment\n")
        f.write("- `0.70-0.75`: Good alignment\n")
        f.write("- `> 0.75`: Excellent alignment ✨\n\n")
        f.write("---\n\n")

        # Overall Summary
        f.write("## 📈 Overall Summary\n\n")

        # Calculate overall grade
        scores = {
            'Semantic Diversity': sem_div['diversity_score'],
            # Penalize deviation from 0.5
            'Realism (balanced)': 1 - abs(real_vs['avg_similarity'] - 0.5) * 2,
            'Sentiment Alignment': sent_align['correlation']
        }

        overall_score = sum(scores.values()) / len(scores)

        f.write("| Metric | Score | Status |\n")
        f.write("|--------|-------|--------|\n")
        f.write(
            f"| Semantic Diversity | {sem_div['diversity_score']:.4f} | {sem_div['interpretation']} |\n")
        f.write(
            f"| Real-vs-Synthetic | {real_vs['avg_similarity']:.4f} | {real_vs['interpretation']} |\n")
        f.write(
            f"| Sentiment Alignment | {sent_align['correlation']:.4f} | {sent_align['interpretation']} |\n")
        f.write(f"| **Overall Score** | **{overall_score:.4f}** | - |\n\n")

        # Final grade
        if overall_score >= 0.75:
            grade = "A"
            emoji = "🌟"
        elif overall_score >= 0.65:
            grade = "B"
            emoji = "✅"
        elif overall_score >= 0.55:
            grade = "C"
            emoji = "⚠️"
        elif overall_score >= 0.45:
            grade = "D"
            emoji = "⚠️"
        else:
            grade = "F"
            emoji = "❌"

        f.write(f"### Final Grade: {emoji} **{grade}**\n\n")

        f.write("---\n\n")
        f.write(
            f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")

    print(f"✓ Saved markdown report: QUALITY_REPORT.md")

    return output_dir, json_path, md_path


def export_reviews_to_folder(output_dir: str, model_name: str = None):
    """
    Export all reviews from database to CSV in the output folder

    Args:
        output_dir: Directory to save the reviews
        model_name: Optional model name to filter reviews

    Returns:
        Path to the saved CSV file
    """
    import os

    # Load reviews from database
    db = initialize_database()
    reviews = db.get_all_reviews(model_name=model_name)
    db.close()

    # Convert to DataFrame and save
    df = pd.DataFrame(reviews)
    csv_path = os.path.join(output_dir, "generated_reviews.csv")
    df.to_csv(csv_path, index=False)

    print(f"✓ Saved {len(df)} reviews: generated_reviews.csv")
    return csv_path


if __name__ == "__main__":
    # Get model name from config
    model_name = CONFIG.get('model_configuration', {}).get(
        'model_name', 'unknown_model')

    # If multiple models, use first one
    if isinstance(model_name, list):
        model_name = model_name[0]

    # Calculate all metrics
    results = calculate_all_metrics(model_name=model_name)

    # Save results and create output folder
    output_dir, json_path, md_path = save_results(results, model_name)

    # Export reviews to the same folder
    reviews_csv = export_reviews_to_folder(output_dir, model_name=model_name)

    print("\n" + "=" * 70)
    print("✅ ALL OUTPUTS SAVED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Output Directory: {output_dir}")
    print(f"   📊 Quality Report: QUALITY_REPORT.md")
    print(f"   📄 Reviews CSV: generated_reviews.csv")
    print(f"   📋 Metrics JSON: quality_metrics.json")
    print("=" * 70)
