from typing import Literal

from pydantic import BaseModel, Field


class ImbalanceRecommendation(BaseModel):

    target_column: str = Field(
        description="Name of the target column."
    )

    problem_type: Literal[
        "Classification",
        "Regression",
        "Unknown"
    ]

    action: Literal[
        "No Action",
        "Class Weights",
        "SMOTE",
        "Random Undersampling"
    ]

    reason: str = Field(
        description=(
            "Reason for the recommended imbalance "
            "handling strategy."
        )
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ]


class ImbalanceRecommendations(BaseModel):

    recommendations: list[
        ImbalanceRecommendation
    ]