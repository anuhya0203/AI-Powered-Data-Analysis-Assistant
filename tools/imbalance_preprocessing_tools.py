import pandas as pd

from sklearn.utils.class_weight import (
    compute_class_weight
)

from imblearn.over_sampling import (
    SMOTE
)

from imblearn.under_sampling import (
    RandomUnderSampler
)


def apply_class_weights(
    df,
    target_column
):

    target = df[
        target_column
    ].dropna()

    classes = target.unique()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target
    )

    class_weights = {
        str(cls): float(weight)
        for cls, weight
        in zip(classes, weights)
    }

    return df.copy(), class_weights


def apply_smote(
    df,
    target_column
):

    if target_column not in df.columns:
        raise ValueError(
            "Target column does not exist."
        )

    working_df = df.dropna(
        subset=[target_column]
    ).copy()

    X = working_df.drop(
        columns=[target_column]
    )

    y = working_df[
        target_column
    ]

    # SMOTE requires numerical features.
    X = pd.get_dummies(
        X,
        drop_first=False
    )

    # Fill any remaining missing values
    X = X.fillna(0)

    smote = SMOTE(
        random_state=42
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X,
            y
        )
    )

    result = pd.DataFrame(
        X_resampled,
        columns=X.columns
    )

    result[target_column] = (
        y_resampled.values
    )

    return result


def apply_random_undersampling(
    df,
    target_column
):

    if target_column not in df.columns:
        raise ValueError(
            "Target column does not exist."
        )

    working_df = df.dropna(
        subset=[target_column]
    ).copy()

    X = working_df.drop(
        columns=[target_column]
    )

    y = working_df[
        target_column
    ]

    X = pd.get_dummies(
        X,
        drop_first=False
    )

    X = X.fillna(0)

    sampler = RandomUnderSampler(
        random_state=42
    )

    X_resampled, y_resampled = (
        sampler.fit_resample(
            X,
            y
        )
    )

    result = pd.DataFrame(
        X_resampled,
        columns=X.columns
    )

    result[target_column] = (
        y_resampled.values
    )

    return result


def apply_imbalance_action(
    df,
    target_column,
    action
):

    if action == "No Action":

        return df.copy(), None


    if action == "Class Weights":

        return apply_class_weights(
            df,
            target_column
        )


    if action == "SMOTE":

        result = apply_smote(
            df,
            target_column
        )

        return result, None


    if action == "Random Undersampling":

        result = apply_random_undersampling(
            df,
            target_column
        )

        return result, None


    raise ValueError(
        f"Unknown imbalance action: {action}"
    )