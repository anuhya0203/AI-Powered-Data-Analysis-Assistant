from typing import Literal

from pydantic import BaseModel, Field


class SchemaRecommendation(BaseModel):

    column: str = Field(
        description="Name of the column."
    )

    action: Literal[
        "No Action",
        "Convert to Numeric",
        "Convert to Datetime",
        "Convert to Categorical",
        "Drop Column",
        "Treat as Identifier"
    ] = Field(
        description="The single recommended schema action."
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


class SchemaDuplicateRecommendation(BaseModel):

    action: Literal[
        "Remove Duplicates",
        "Keep Duplicates"
    ] = Field(
        description="The single recommended action for exact duplicate rows."
    )

    reason: str = Field(
        description="Reason for the duplicate recommendation."
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ] = Field(
        description="Confidence in the duplicate recommendation."
    )


class SchemaRecommendations(BaseModel):

    duplicate_recommendation: SchemaDuplicateRecommendation = Field(
        description="Dataset-level recommendation for handling exact duplicate rows."
    )

    recommendations: list[
        SchemaRecommendation
    ]
