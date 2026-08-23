from langchain_core.tools import tool

from tools.outlier_tools import analyze_outlier_columns


def create_outlier_tools(df, selected_columns):

    @tool
    def analyze_selected_outliers() -> list:
        """
        Analyze the selected numeric columns for outliers.

        Returns statistical evidence including:
        outlier count, outlier percentage, skewness,
        IQR boundaries, mean, standard deviation,
        and Z-score outlier count.

        Use this information to determine the most
        appropriate outlier treatment strategy.
        """

        return analyze_outlier_columns(
            df,
            selected_columns
        )

    return [
        analyze_selected_outliers
    ]