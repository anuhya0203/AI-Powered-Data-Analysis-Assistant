from typing import Literal

from pydantic import BaseModel, Field


class FeatureRelevanceRecommendation(
    BaseModel
):

    column: str = Field(
        description="Name of the feature."
    )

    action: Literal[
        "Keep",
        "Drop",
        "Investigate"
    ] = Field(
        description=(
            "Recommended action for the feature."
        )
    )

    reason: str = Field(
        description=(
            "Reason for the recommendation."
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


class FeatureRelevanceRecommendations(
    BaseModel
):

    recommendations: list[
        FeatureRelevanceRecommendation
    ]