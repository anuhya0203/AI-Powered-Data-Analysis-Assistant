from langchain_core.tools import tool

from tools.duplicate_tools import analyze_duplicates


def create_duplicate_tools(df):

    @tool
    def analyze_dataset_duplicates() -> dict:
        """
        Analyze duplicate rows and possible identifier
        columns in the current dataset.

        Returns duplicate counts, duplicate percentage,
        duplicate groups, and possible identifier columns.
        """

        return analyze_duplicates(
            df
        )

    return [
        analyze_dataset_duplicates
    ]