from typing import Literal

from pydantic import BaseModel, Field


class DuplicateRecommendation(BaseModel):

    action: Literal[
        "Remove Duplicates",
        "Keep Duplicates"
    ] = Field(
        description=(
            "The recommended action for duplicate rows."
        )
    )

    reason: str = Field(
        description=(
            "Statistical and data-quality reason "
            "for the recommendation."
        )
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ] = Field(
        description=(
            "Confidence in the recommendation."
        )
    )


class DuplicateRecommendations(BaseModel):

    recommendation: DuplicateRecommendation