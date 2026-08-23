import pandas as pd

from tools.profiling_tools import profile_dataset
from tools.missingness_tools import (
    analyze_numeric_column,
    analyze_categorical_column,
    analyze_missingness_by_category,
)


class MissingValueAgent:

    def __init__(self, df):
        self.df = df

        self.profile = None
        self.observations = {}
        self.recommendations = {}

    def inspect_dataset(self):
        self.profile = profile_dataset(self.df)
        return self.profile

    def inspect_column(self, column):

        if self.profile is None:
            self.inspect_dataset()

        column_info = next(
            item
            for item in self.profile["columns_info"]
            if item["column"] == column
        )

        if column_info["missing_count"] == 0:
            return {
                "column": column,
                "message": "No missing values."
            }

        if pd.api.types.is_numeric_dtype(
            self.df[column]
        ):
            result = analyze_numeric_column(
                self.df,
                column
            )
        else:
            result = analyze_categorical_column(
                self.df,
                column
            )

        self.observations[column] = result

        return result

    def get_missing_columns(self):

        if self.profile is None:
            self.inspect_dataset()

        return [
            item["column"]
            for item in self.profile["columns_info"]
            if item["missing_count"] > 0
        ]

    def inspect_missingness_relationship(
        self,
        target_column,
        grouping_column
    ):

        result = analyze_missingness_by_category(
            self.df,
            target_column,
            grouping_column
        )

        self.observations[
            f"{target_column}_by_{grouping_column}"
        ] = result

        return result