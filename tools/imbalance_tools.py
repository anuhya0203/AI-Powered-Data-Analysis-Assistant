import pandas as pd
import numpy as np


def analyze_target_distribution(
    df,
    target_column
):
    """
    Analyze the target column for class imbalance.
    """

    if target_column is None:
        return {
            "target_column": None,
            "problem_type": "Unknown",
            "class_count": 0,
            "class_distribution": {},
            "minority_class": None,
            "majority_class": None,
            "minority_count": 0,
            "majority_count": 0,
            "imbalance_ratio": None,
            "minority_percentage": None
        }

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    series = df[target_column].dropna()

    if len(series) == 0:
        return {
            "target_column": target_column,
            "problem_type": "Unknown",
            "class_count": 0,
            "class_distribution": {},
            "minority_class": None,
            "majority_class": None,
            "minority_count": 0,
            "majority_count": 0,
            "imbalance_ratio": None,
            "minority_percentage": None
        }

    unique_count = series.nunique()

    # --------------------------------------------------
    # Determine whether this looks like classification
    # --------------------------------------------------

    if (
        pd.api.types.is_object_dtype(series)
        or
        pd.api.types.is_string_dtype(series)
        or
        pd.api.types.is_bool_dtype(series)
        or
        isinstance(
            series.dtype,
            pd.CategoricalDtype
        )
    ):

        problem_type = "Classification"

    elif unique_count <= 10:

        # Small number of discrete numeric values
        # may represent classes.
        problem_type = "Classification"

    else:

        problem_type = "Regression"


    # --------------------------------------------------
    # Regression does not have class imbalance
    # --------------------------------------------------

    if problem_type == "Regression":

        return {
            "target_column": target_column,
            "problem_type": "Regression",
            "class_count": unique_count,
            "class_distribution": {},
            "minority_class": None,
            "majority_class": None,
            "minority_count": 0,
            "majority_count": 0,
            "imbalance_ratio": None,
            "minority_percentage": None
        }


    # --------------------------------------------------
    # Class distribution
    # --------------------------------------------------

    counts = (
        series
        .value_counts()
    )

    class_distribution = {
        str(key): int(value)
        for key, value
        in counts.items()
    }


    majority_count = int(
        counts.iloc[0]
    )

    minority_count = int(
        counts.iloc[-1]
    )

    majority_class = str(
        counts.index[0]
    )

    minority_class = str(
        counts.index[-1]
    )


    # --------------------------------------------------
    # Imbalance ratio
    # --------------------------------------------------

    imbalance_ratio = (
        majority_count
        / minority_count
        if minority_count > 0
        else None
    )


    minority_percentage = (
        minority_count
        / len(series)
        * 100
    )


    return {

        "target_column": target_column,

        "problem_type": problem_type,

        "class_count": int(
            unique_count
        ),

        "class_distribution":
            class_distribution,

        "minority_class":
            minority_class,

        "majority_class":
            majority_class,

        "minority_count":
            minority_count,

        "majority_count":
            majority_count,

        "imbalance_ratio":
            round(
                imbalance_ratio,
                2
            )
            if imbalance_ratio is not None
            else None,

        "minority_percentage":
            round(
                minority_percentage,
                2
            )
    }