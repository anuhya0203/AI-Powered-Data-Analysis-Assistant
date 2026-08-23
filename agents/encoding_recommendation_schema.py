from typing import Literal

from pydantic import BaseModel, Field


class EncodingRecommendation(BaseModel):

    column: str = Field(
        description="Name of the categorical column."
    )

    action: Literal[
        "No Action",
        "One-Hot Encoding",
        "Frequency Encoding",
        "Target Encoding",
        "Ordinal Encoding"
    ] = Field(
        description=(
            "The single recommended encoding strategy."
        )
    )

    reason: str = Field(
        description=(
            "Reason for selecting this encoding strategy."
        )
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ] = Field(
        description=(
            "Confidence in the encoding recommendation."
        )
    )


class EncodingRecommendations(BaseModel):

    recommendations: list[
        EncodingRecommendation
    ]