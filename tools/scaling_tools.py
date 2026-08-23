import pandas as pd
import numpy as np


def analyze_scaling_column(df, column):

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    count = len(series)

    if count == 0:
        return {
            "column": column,
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "skewness": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "outlier_percentage": 0.0
        }

    mean = float(series.mean())
    std = float(series.std())

    min_value = float(series.min())
    max_value = float(series.max())

    skewness = float(series.skew())

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))

    iqr = q3 - q1

    # IQR outlier detection
    if iqr == 0:

        outlier_percentage = 0.0

    else:

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        outlier_percentage = (
            outliers.sum()
            / count
            * 100
        )

    return {

        "column": column,

        "count": count,

        "mean": round(mean, 4),

        "std": round(std, 4),

        "min": round(min_value, 4),

        "max": round(max_value, 4),

        "range": round(
            max_value - min_value,
            4
        ),

        "skewness": round(
            skewness,
            4
        ),

        "q1": round(q1, 4),

        "q3": round(q3, 4),

        "iqr": round(iqr, 4),

        "outlier_percentage": round(
            outlier_percentage,
            2
        )
    }


def analyze_scaling_features(
    df,
    target_column=None
):

    results = []

    for column in df.columns:

        # Never scale the target here
        if column == target_column:
            continue

        series = df[column]

        # Only numerical columns
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # Exclude obvious temporal columns
        if column.lower() in {
            "year",
            "date",
            "month",
            "day",
            "timestamp"
        }:
            continue

        results.append(
            analyze_scaling_column(
                df,
                column
            )
        )

    return results