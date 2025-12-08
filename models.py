"""
Pydantic models for structured review generation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List
from enum import IntEnum

class Rating(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class Review(BaseModel):
    rating: Rating = Field(..., description="Star rating from 1 to 5")
    review_text: str = Field(..., min_length=10, max_length=2000)


class ReviewBatch(BaseModel):
    reviews: List[Review] = Field(
        ...,
        description="List of reviews"
    )



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
