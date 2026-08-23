import pandas as pd
import numpy as np


def analyze_outlier_column(df, column):
    """
    Analyze a numeric column for outliers.

    Python calculates the statistical evidence.
    The LLM will later use this evidence to choose
    the appropriate preprocessing strategy.
    """

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return {
            "column": column,
            "count": 0,
            "outliers": 0,
            "outlier_percentage": 0.0,
            "skewness": None,
            "q1": None,
            "q3": None,
            "lower_bound": None,
            "upper_bound": None,
            "mean": None,
            "std": None
        }

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # --------------------------------------------------
    # IQR outliers
    # --------------------------------------------------

    outlier_mask = (
        (series < lower_bound)
        |
        (series > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    outlier_percentage = (
        outlier_count / len(series)
    ) * 100

    # --------------------------------------------------
    # Skewness
    # --------------------------------------------------

    skewness = series.skew()

    if pd.isna(skewness):
        skewness = 0.0

    # --------------------------------------------------
    # Z-score information
    # --------------------------------------------------

    mean = series.mean()
    std = series.std()

    if std != 0 and not pd.isna(std):

        z_scores = (
            (series - mean) / std
        )

        zscore_outliers = int(
            (z_scores.abs() > 3).sum()
        )

    else:

        zscore_outliers = 0

    # --------------------------------------------------
    # Return evidence
    # --------------------------------------------------

    return {
        "column": column,
        "count": int(len(series)),

        "outliers": outlier_count,

        "outlier_percentage": round(
            float(outlier_percentage),
            2
        ),

        "skewness": round(
            float(skewness),
            4
        ),

        "q1": round(
            float(q1),
            4
        ),

        "q3": round(
            float(q3),
            4
        ),

        "lower_bound": round(
            float(lower_bound),
            4
        ),

        "upper_bound": round(
            float(upper_bound),
            4
        ),

        "mean": round(
            float(mean),
            4
        ),

        "std": round(
            float(std),
            4
        ),

        "zscore_outliers": zscore_outliers
    }


def analyze_outlier_columns(df, columns):
    """
    Analyze multiple numeric columns.
    """

    results = []

    for column in columns:

        results.append(
            analyze_outlier_column(
                df,
                column
            )
        )

    return results