from typing import Literal

from pydantic import BaseModel, Field


class OutlierRecommendation(BaseModel):

    column: str = Field(
        description="Name of the numeric column."
    )

    action: Literal[
        "IQR",
        "Winsorization",
        "Z-Score",
        "Remove Rows",
        "No Action"
    ] = Field(
        description="The single best outlier treatment strategy."
    )

    reason: str = Field(
        description="Statistical reason for selecting this strategy."
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ] = Field(
        description="Confidence in the recommendation."
    )


class OutlierRecommendations(BaseModel):

    recommendations: list[
        OutlierRecommendation
    ]