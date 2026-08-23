from typing import Literal
from pydantic import BaseModel, Field


class ColumnRecommendation(BaseModel):
    column: str = Field(
        description="Name of the column containing missing values."
    )

    action: Literal[
        "Mean",
        "Median",
        "Mode",
        "Drop Column",
        "Leave Missing",
        "No Action"
    ] = Field(
        description="The single best preprocessing action."
    )

    reason: str = Field(
        description="Why this action is appropriate."
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ] = Field(
        description="Confidence in the recommendation."
    )


class MissingValueRecommendations(BaseModel):
    recommendations: list[ColumnRecommendation]