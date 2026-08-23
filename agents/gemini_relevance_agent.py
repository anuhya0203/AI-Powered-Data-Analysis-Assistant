from langchain_core.tools import tool

from tools.relevance_tools import (
    analyze_feature_relevance
)


def create_relevance_tools(
    df,
    target_column,
    schema_analysis
):

    @tool
    def analyze_feature_relevance_tool() -> dict:
        """
        Analyze feature relevance using existing
        Schema Agent evidence plus correlation analysis.
        """

        return analyze_feature_relevance(
            df=df,
            target_column=target_column,
            schema_analysis=schema_analysis
        )

    return [
        analyze_feature_relevance_tool
    ]