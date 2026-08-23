import pandas as pd


def analyze_duplicates(df):
    """
    Analyze duplicate rows in the dataset.

    Python calculates the evidence.
    The LLM will later decide whether duplicate
    rows should be removed or retained.
    """

    total_rows = len(df)

    duplicate_mask = df.duplicated(
        keep=False
    )

    duplicate_rows = df[duplicate_mask]

    duplicate_count = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_count / total_rows * 100
        if total_rows > 0
        else 0
    )

    duplicate_groups = (
        duplicate_rows
        .value_counts()
        .reset_index(
            name="count"
        )
    )

    # --------------------------------------------------
    # Identify columns that may act as identifiers
    # --------------------------------------------------

    identifier_candidates = []

    for column in df.columns:

        unique_count = df[column].nunique(
            dropna=False
        )

        unique_percentage = (
            unique_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        # A column with nearly all unique values
        # may be an identifier.
        if unique_percentage >= 95:

            identifier_candidates.append({
                "column": column,
                "unique_values": int(
                    unique_count
                ),
                "unique_percentage": round(
                    unique_percentage,
                    2
                )
            })

    # --------------------------------------------------
    # Return statistical evidence
    # --------------------------------------------------

    return {
        "total_rows": total_rows,

        "duplicate_rows": duplicate_count,

        "duplicate_percentage": round(
            duplicate_percentage,
            2
        ),

        "duplicate_groups": (
            duplicate_groups.to_dict(
                orient="records"
            )
        ),

        "identifier_candidates": (
            identifier_candidates
        )
    }