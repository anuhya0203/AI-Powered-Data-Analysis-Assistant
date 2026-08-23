from langchain_core.tools import tool

from tools.schema_tools import analyze_schema


def create_schema_tools(df):

    @tool
    def analyze_dataset_schema() -> dict:
        """
        Analyze the dataset schema and duplicate-row characteristics.

        Returns dtype information, missingness, uniqueness,
        numeric conversion potential, date conversion potential,
        identifier likelihood, constant-column status, sample values,
        and deterministic duplicate-row statistics.
        """

        return analyze_schema(df)

    return [
        analyze_dataset_schema
    ]