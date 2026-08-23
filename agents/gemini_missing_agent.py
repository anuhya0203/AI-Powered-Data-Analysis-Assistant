from langchain_core.tools import tool

from tools.profiling_tools import profile_dataset
from tools.missingness_tools import (
    analyze_numeric_column,
    analyze_categorical_column,
    analyze_missingness_by_category,
)


def create_missing_value_tools(df):

    @tool
    def profile_dataset_tool() -> dict:
        """
        Profile the dataset and return information about
        rows, columns, missing values, data types,
        and unique values.
        """
        return profile_dataset(df)

    @tool
    def analyze_numeric_column_tool(
        column: str
    ) -> dict:
        """
        Analyze a numeric column. Returns mean, median,
        standard deviation, minimum, maximum, skewness,
        and quartiles.
        """
        return analyze_numeric_column(
            df,
            column
        )

    @tool
    def analyze_categorical_column_tool(
        column: str
    ) -> dict:
        """
        Analyze a categorical column. Returns number of
        unique values, most frequent value, and
        top categories.
        """
        return analyze_categorical_column(
            df,
            column
        )

    @tool
    def analyze_missingness_by_category_tool(
        target_column: str,
        grouping_column: str
    ) -> list:
        """
        Analyze whether missingness in one column differs
        across groups of another column.
        """
        return analyze_missingness_by_category(
            df,
            target_column,
            grouping_column
        )

    return [
        profile_dataset_tool,
        analyze_numeric_column_tool,
        analyze_categorical_column_tool,
        analyze_missingness_by_category_tool,
    ]