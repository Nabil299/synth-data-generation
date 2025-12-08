"""
Quality Guardrails for Synthetic Review Generation

This module handles all quality validation including:
- Duplicate detection via semantic similarity
- Sentiment-rating mismatch detection
- Future: Vocabulary diversity, domain realism, etc.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class QualityGuardrails:
    def __init__(self, db, config: dict, model_name: str = None):
        """
        Initialize quality guardrails

        Args:
            db: Database instance (for querying existing reviews only)
            config: Configuration dictionary from config.yaml
            model_name: Optional model name to filter duplicates by same model only
        """
        self.db = db
        self.config = config.get('quality_guardrails', {})
        self.model_name = model_name

        # Initialize embedding model for duplicate detection
        embedding_model_name = self.config.get(
            'embedding_model', 'all-MiniLM-L6-v2')
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded (dimension: {self.embedding_dim})")

        # Initialize sentiment analyzer
        self.enable_sentiment_check = self.config.get(
            'enable_sentiment_check', True)
        if self.enable_sentiment_check:
            print("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=-1  # CPU, use 0 for GPU
            )
            print("✓ Sentiment model loaded")

        # Duplicate detection settings
        self.enable_duplicate_check = self.config.get(
            'enable_duplicate_check', True)
        self.similarity_threshold = self.config.get(
            'similarity_threshold', 0.85)
        self.min_reviews_before_check = self.config.get(
            'min_reviews_before_check', 30)

        # Sentiment thresholds
        self.sentiment_thresholds = self.config.get('sentiment_thresholds', {
            'rating_1': {'min': 0.0, 'max': 0.3},
            'rating_2': {'min': 0.0, 'max': 0.4},
            'rating_3': {'min': 0.3, 'max': 0.7},
            'rating_4': {'min': 0.6, 'max': 1.0},
            'rating_5': {'min': 0.7, 'max': 1.0}
        })

        print(f"\n✓ Quality Guardrails initialized")
        print(
            f"  - Duplicate check: {self.enable_duplicate_check} (threshold: {self.similarity_threshold})")
        print(f"  - Sentiment check: {self.enable_sentiment_check}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Text to embed

        Returns:
            Numpy array of embeddings
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding

    def check_duplicate(
        self,
        review_text: str,
        embedding: np.ndarray
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """
        Check if a review is a duplicate using semantic similarity

        Args:
            review_text: The review text to check
            embedding: Pre-computed embedding of the review

        Returns:
            Tuple of (is_duplicate, duplicate_of_id, similarity_score)
        """
        if not self.enable_duplicate_check:
            return False, None, None

        # Check if we have enough reviews to start checking (filtered by model if specified)
        review_count = self.db.get_review_count(model_name=self.model_name)
        if review_count < self.min_reviews_before_check:
            return False, None, None

        # Find most similar review (filtered by model if specified)
        similar_reviews = self.db.find_similar_reviews(
            embedding, limit=1, model_name=self.model_name
        )

        if similar_reviews:
            review_id, similarity, similar_text = similar_reviews[0]
            if similarity >= self.similarity_threshold:
                return True, review_id, similarity

        return False, None, None

    def check_sentiment_mismatch(
        self,
        review_text: str,
        rating: int
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Check if sentiment matches the rating

        Args:
            review_text: The review text
            rating: Rating value (1-5)

        Returns:
            Tuple of (is_mismatch, sentiment_score, reason)
        """
        if not self.enable_sentiment_check:
            return False, None, None

        try:
            # Get sentiment prediction
            result = self.sentiment_analyzer(review_text[:512])[
                0]

            # Convert cardiffnlp/twitter-roberta sentiment to score (0=negative, 1=positive)
            # This model outputs: LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)
            label = result['label']
            confidence = result['score']

            if label == 'LABEL_0':  # Negative
                sentiment_score = (1 - confidence) * 0.5  # Map to 0-0.5 range
            elif label == 'LABEL_1':  # Neutral
                sentiment_score = 0.5  # Map to middle
            elif label == 'LABEL_2':  # Positive
                # Map to 0.5-1.0 range
                sentiment_score = 0.5 + (confidence * 0.5)
            else:
                # Fallback for unexpected labels
                sentiment_score = 0.5

            # Get expected range for this rating
            threshold_key = f'rating_{rating}'
            if threshold_key not in self.sentiment_thresholds:
                return False, sentiment_score, None

            expected = self.sentiment_thresholds[threshold_key]
            min_expected = expected['min']
            max_expected = expected['max']

            # Check if sentiment is within expected range
            if sentiment_score < min_expected:
                reason = f"Sentiment too negative ({sentiment_score:.2f}) for {rating}-star rating (expected >= {min_expected})"
                return True, sentiment_score, reason
            elif sentiment_score > max_expected:
                reason = f"Sentiment too positive ({sentiment_score:.2f}) for {rating}-star rating (expected <= {max_expected})"
                return True, sentiment_score, reason

            return False, sentiment_score, None

        except Exception as e:
            print(f"Warning: Sentiment check failed: {e}")
            return False, None, None

    def validate_review(
        self,
        rating: int,
        review_text: str
    ) -> Dict[str, any]:
        """
        Run all quality checks on a review

        Args:
            rating: Rating value (1-5)
            review_text: Review text

        Returns:
            Dictionary with validation results:
            {
                'passed': bool,
                'embedding': np.ndarray,
                'flags': List[str],  # List of issues found
                'scores': {
                    'sentiment': float,
                    'similarity': float
                },
                'rejection_reason': str or None
            }
        """
        result = {
            'passed': True,
            'embedding': None,
            'flags': [],
            'scores': {},
            'rejection_reason': None
        }

        # Generate embedding (needed for both duplicate check and storage)
        embedding = self.embed_text(review_text)
        result['embedding'] = embedding

        # Check 1: Duplicate detection
        if self.enable_duplicate_check:
            is_dup, dup_id, similarity = self.check_duplicate(
                review_text, embedding)
            if similarity is not None:
                result['scores']['similarity'] = float(similarity)

            if is_dup:
                result['passed'] = False
                result['flags'].append('duplicate')
                result['rejection_reason'] = f"Duplicate (similarity: {similarity:.3f} with review #{dup_id})"
                return result  # Early return, no need to check further

        # Check 2: Sentiment-rating mismatch
        if self.enable_sentiment_check:
            is_mismatch, sentiment_score, reason = self.check_sentiment_mismatch(
                review_text, rating
            )

            if sentiment_score is not None:
                result['scores']['sentiment'] = float(sentiment_score)

            if is_mismatch:
                result['passed'] = False
                result['flags'].append('sentiment_mismatch')
                result['rejection_reason'] = reason
                return result

        return result

    def validate_batch(
        self,
        reviews: List[Dict[str, any]],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Validate a batch of reviews

        Args:
            reviews: List of review dicts with 'rating' and 'review_text'

        Returns:
            Tuple of (accepted_reviews, rejected_reviews, stats)
        """
        accepted = []
        rejected = []
        stats = {
            'total': len(reviews),
            'accepted': 0,
            'duplicates': 0,
            'sentiment_mismatches': 0,
            'other_rejections': 0
        }

        for review in reviews:
            validation = self.validate_review(
                rating=review['rating'],
                review_text=review['review_text']
            )

            # Add validation results to review
            review_with_validation = {
                **review,
                'embedding': validation['embedding'],
                'sentiment_score': validation['scores'].get('sentiment'),
                'similarity_score': validation['scores'].get('similarity'),
                'quality_flags': validation['flags']
            }

            if validation['passed']:
                accepted.append(review_with_validation)
                stats['accepted'] += 1
            else:
                review_with_validation['rejection_reason'] = validation['rejection_reason']
                rejected.append(review_with_validation)

                # Track rejection reasons
                if 'duplicate' in validation['flags']:
                    stats['duplicates'] += 1
                elif 'sentiment_mismatch' in validation['flags']:
                    stats['sentiment_mismatches'] += 1
                else:
                    stats['other_rejections'] += 1

        return accepted, rejected, stats
