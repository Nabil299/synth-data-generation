"""
Pydantic models for structured review generation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List
from enum import IntEnum


class Rating(IntEnum):
    """Star rating from 1 to 5"""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class Review(BaseModel):
    """Single review model"""
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    review_text: str = Field(..., min_length=10, max_length=2000,
                             description="The review text content")

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if v not in [1, 2, 3, 4, 5]:
            raise ValueError('Rating must be between 1 and 5')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "rating": 5,
                "review_text": "This product is amazing! I highly recommend it to everyone."
            }
        }


class ReviewBatch(BaseModel):
    """Batch of reviews with numbered keys"""
    reviews: Dict[str, Review] = Field(
        ...,
        description="Dictionary of reviews with string keys ('1', '2', '3', etc.)"
    )

    @field_validator('reviews')
    @classmethod
    def validate_review_count(cls, v):
        # This validation can be customized based on expected batch size
        if len(v) < 1:
            raise ValueError('Must contain at least one review')
        return v

    def to_list(self) -> List[Review]:
        """Convert to a list of reviews"""
        return [self.reviews[key] for key in sorted(self.reviews.keys(), key=lambda x: int(x))]

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": {
                    "1": {"rating": 5, "review_text": "Great product!"},
                    "2": {"rating": 4, "review_text": "Good quality."},
                    "3": {"rating": 3, "review_text": "Average experience."}
                }
            }
        }


def parse_review_batch(json_response: str) -> ReviewBatch:
    """
    Parse a JSON response into a ReviewBatch object

    Args:
        json_response: JSON string response from the API

    Returns:
        Validated ReviewBatch object

    Raises:
        ValidationError: If the response doesn't match the schema
    """
    import json
    data = json.loads(json_response)

    # Handle both formats: direct reviews dict or nested under "reviews" key
    if "reviews" in data:
        return ReviewBatch(**data)
    else:
        # Wrap in reviews key
        return ReviewBatch(reviews=data)
