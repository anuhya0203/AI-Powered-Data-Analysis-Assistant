import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


def apply_scaler(
    df,
    column,
    scaler
):

    result = df.copy()

    values = result[
        [column]
    ].copy()

    transformed = scaler.fit_transform(
        values
    )

    result[column] = transformed.flatten()

    return result


def apply_scaling_action(
    df,
    column,
    action
):

    if action == "No Action":

        return df.copy()

    elif action == "StandardScaler":

        return apply_scaler(
            df,
            column,
            StandardScaler()
        )

    elif action == "MinMaxScaler":

        return apply_scaler(
            df,
            column,
            MinMaxScaler()
        )

    elif action == "RobustScaler":

        return apply_scaler(
            df,
            column,
            RobustScaler()
        )

    else:

        raise ValueError(
            f"Unknown scaling action: {action}"
        )


def apply_scaling_actions(
    df,
    selected_actions
):

    result = df.copy()

    summary = []

    for column, action in selected_actions.items():

        if column not in result.columns:
            continue

        before_min = result[
            column
        ].min()

        before_max = result[
            column
        ].max()

        result = apply_scaling_action(
            result,
            column,
            action
        )

        after_min = result[
            column
        ].min()

        after_max = result[
            column
        ].max()

        summary.append({

            "Column": column,

            "Action": action,

            "Before Min": round(
                float(before_min),
                4
            ),

            "Before Max": round(
                float(before_max),
                4
            ),

            "After Min": round(
                float(after_min),
                4
            ),

            "After Max": round(
                float(after_max),
                4
            )

        })

    return result, summary