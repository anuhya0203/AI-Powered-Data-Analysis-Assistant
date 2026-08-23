import pandas as pd
import numpy as np


def apply_iqr(df, column):
    """
    Remove rows containing IQR outliers.
    """

    result = df.copy()

    series = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
        series >= lower_bound
    ) & (
        series <= upper_bound
    )

    # Keep NaN rows because this tool is only
    # responsible for outliers.
    mask = mask | series.isna()

    return result.loc[mask].copy()


def apply_winsorization(df, column):
    """
    Cap outliers at the IQR lower and upper bounds.
    """

    result = df.copy()

    series = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    result[column] = series.clip(
        lower=lower_bound,
        upper=upper_bound
    )

    return result


def apply_zscore(df, column, threshold=3):
    """
    Remove rows whose absolute Z-score exceeds
    the specified threshold.
    """

    result = df.copy()

    series = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    mean = series.mean()
    std = series.std()

    # If standard deviation is zero or unavailable,
    # there are no meaningful Z-score outliers.
    if pd.isna(std) or std == 0:

        return result

    z_scores = (
        (series - mean) / std
    )

    mask = (
        z_scores.abs() <= threshold
    )

    # Preserve missing values.
    mask = mask | series.isna()

    return result.loc[mask].copy()


def apply_remove_rows(df, column):
    """
    Remove rows identified as IQR outliers.

    This is intentionally equivalent to the IQR
    removal strategy for now.
    """

    return apply_iqr(
        df,
        column
    )


def apply_outlier_action(df, column, action):
    """
    Apply one selected outlier action to one column.
    """

    if action == "IQR":

        return apply_iqr(
            df,
            column
        )

    elif action == "Winsorization":

        return apply_winsorization(
            df,
            column
        )

    elif action == "Z-Score":

        return apply_zscore(
            df,
            column
        )

    elif action == "Remove Rows":

        return apply_remove_rows(
            df,
            column
        )

    elif action == "No Action":

        return df.copy()

    else:

        raise ValueError(
            f"Unknown outlier action: {action}"
        )


def apply_outlier_actions(df, selected_actions):
    """
    Apply selected outlier actions to multiple columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Current working dataset.

    selected_actions : dict
        Example:
        {
            "Area": "Winsorization",
            "Rainfall": "IQR",
            "price": "No Action"
        }

    Returns
    -------
    cleaned_df : pandas.DataFrame
    summary : list
    """

    cleaned_df = df.copy()

    summary = []

    for column, action in selected_actions.items():

        if column not in cleaned_df.columns:
            continue

        rows_before = len(cleaned_df)

        values_before = (
            cleaned_df[column].copy()
        )

        cleaned_df = apply_outlier_action(
            cleaned_df,
            column,
            action
        )

        rows_after = len(cleaned_df)

        # ------------------------------------------------
        # Count affected values
        # ------------------------------------------------

        if action == "Winsorization":

            values_after = cleaned_df[column]

            # Compare values that still exist after
            # preprocessing.
            common_index = values_before.index.intersection(
                values_after.index
            )

            changed_values = int(
                (
                    values_before.loc[common_index]
                    != values_after.loc[common_index]
                ).sum()
            )

            affected = changed_values

            affected_label = "values capped"

        else:

            affected = (
                rows_before - rows_after
            )

            affected_label = "rows removed"

        summary.append({

            "Column": column,

            "Action": action,

            "Affected": affected,

            "Description": (
                f"{affected} {affected_label}"
            ),

            "Rows Before": rows_before,

            "Rows After": rows_after

        })

    return cleaned_df, summary