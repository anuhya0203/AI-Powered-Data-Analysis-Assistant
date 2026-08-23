from typing import Literal

from pydantic import BaseModel, Field


class ScalingRecommendation(BaseModel):

    column: str = Field(
        description="Name of the numerical feature."
    )

    action: Literal[
        "No Action",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler"
    ] = Field(
        description="The recommended scaling strategy."
    )

    reason: str = Field(
        description="Reason for the recommendation."
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ] = Field(
        description="Confidence in the recommendation."
    )


class ScalingRecommendations(BaseModel):

    recommendations: list[
        ScalingRecommendation
    ]