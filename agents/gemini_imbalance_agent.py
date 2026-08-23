from langchain_core.tools import tool

from tools.imbalance_tools import (
    analyze_target_distribution
)


def create_imbalance_tools(
    df,
    target_column
):

    @tool
    def analyze_target_distribution_tool() -> dict:
        """
        Analyze the target variable and determine whether
        the classification dataset has class imbalance.
        """

        return analyze_target_distribution(
            df,
            target_column
        )

    return [
        analyze_target_distribution_tool
    ]