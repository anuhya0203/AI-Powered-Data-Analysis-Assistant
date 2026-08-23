from langchain_core.tools import tool

from tools.scaling_tools import (
    analyze_scaling_features
)


def create_scaling_tools(
    df,
    target_column=None
):

    @tool
    def analyze_scaling_features_tool() -> list:
        """
        Analyze numerical features and provide the
        statistical evidence needed to determine
        whether scaling is appropriate.
        """

        return analyze_scaling_features(
            df,
            target_column
        )

    return [
        analyze_scaling_features_tool
    ]