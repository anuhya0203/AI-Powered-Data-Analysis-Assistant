import pandas as pd
import numpy as np


def analyze_column_schema(df, column):
    """
    Analyze the schema characteristics of one column.

    Python calculates the evidence.
    The LLM will later interpret the evidence and
    recommend an appropriate schema action.
    """

    series = df[column]

    total_count = len(series)

    missing_count = int(
        series.isnull().sum()
    )

    non_missing = series.dropna()

    unique_count = int(
        series.nunique(dropna=True)
    )

    unique_percentage = (
        unique_count / len(non_missing) * 100
        if len(non_missing) > 0
        else 0
    )

    dtype = str(series.dtype)

    # --------------------------------------------------
    # Numeric detection
    # --------------------------------------------------

    numeric_conversion = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric_success_count = int(
        numeric_conversion.notna().sum()
    )

    numeric_conversion_percentage = (
        numeric_success_count / len(non_missing) * 100
        if len(non_missing) > 0
        else 0
    )

    # --------------------------------------------------
    # Date detection
    # --------------------------------------------------

    try:

        date_conversion = pd.to_datetime(
            series,
            errors="coerce"
        )

        date_success_count = int(
            date_conversion.notna().sum()
        )

        date_conversion_percentage = (
            date_success_count / len(non_missing) * 100
            if len(non_missing) > 0
            else 0
        )

    except Exception:

        date_conversion_percentage = 0.0

    # --------------------------------------------------
    # Empty column
    # --------------------------------------------------

    is_empty = (
        non_missing.empty
    )

    # --------------------------------------------------
    # Constant column
    # --------------------------------------------------

    is_constant = (
        unique_count <= 1
        and not is_empty
    )

    # --------------------------------------------------
    # Identifier candidate
    # --------------------------------------------------

    is_identifier_candidate = (
        unique_percentage >= 95
        and
        total_count > 0
    )

    # --------------------------------------------------
    # Sample values
    # --------------------------------------------------

    sample_values = (
        non_missing
        .head(5)
        .astype(str)
        .tolist()
    )

    return {

        "column": column,

        "pandas_dtype": dtype,

        "total_count": total_count,

        "missing_count": missing_count,

        "missing_percentage": round(
            (
                missing_count / total_count * 100
                if total_count > 0
                else 0
            ),
            2
        ),

        "unique_count": unique_count,

        "unique_percentage": round(
            unique_percentage,
            2
        ),

        "numeric_conversion_percentage": round(
            numeric_conversion_percentage,
            2
        ),

        "date_conversion_percentage": round(
            date_conversion_percentage,
            2
        ),

        "is_empty": is_empty,

        "is_constant": is_constant,

        "is_identifier_candidate": (
            is_identifier_candidate
        ),

        "sample_values": sample_values
    }


def analyze_schema(df):
    """
    Analyze the schema of every column.
    """

    results = []

    for column in df.columns:

        results.append(
            analyze_column_schema(
                df,
                column
            )
        )

    return results