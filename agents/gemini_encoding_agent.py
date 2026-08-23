from langchain_core.tools import tool

from tools.encoding_tools import (
    analyze_categorical_features
)


def create_encoding_tools(
    df,
    target_column=None
):

    @tool
    def analyze_categorical_features_tool() -> list:
        """
        Analyze categorical features and provide the
        statistical evidence required to select an
        appropriate encoding strategy.

        The target column is excluded from encoding.
        """

        return analyze_categorical_features(
            df,
            target_column
        )

    return [
        analyze_categorical_features_tool
    ]