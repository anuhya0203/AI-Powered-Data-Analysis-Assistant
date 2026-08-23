import pandas as pd
import numpy as np


def analyze_feature_relevance(
    df,
    target_column=None,
    schema_analysis=None
):
    """
    Analyze feature relevance using existing Schema Agent
    evidence plus feature-to-feature and feature-to-target
    correlations.

    The Schema Agent evidence is reused rather than
    recalculated.
    """

    if schema_analysis is None:
        raise ValueError(
            "Schema analysis is required for feature relevance analysis."
        )

    # ------------------------------------------------------
    # Normalize Schema Agent output
    # ------------------------------------------------------

    if isinstance(schema_analysis, dict):

        schema_columns = schema_analysis.get(
            "columns",
            []
        )

    elif isinstance(schema_analysis, list):

        schema_columns = schema_analysis

    else:

        raise TypeError(
            "Unexpected schema_analysis type: "
            f"{type(schema_analysis)}"
        )

    schema_features = {}

    for item in schema_columns:

        # Pydantic object
        if hasattr(item, "model_dump"):

            item = item.model_dump()

        # Dictionary
        if isinstance(item, dict):

            column = item.get("column")

            if column is not None:

                schema_features[column] = item


    results = []


    # ======================================================
    # Analyze every feature
    # ======================================================

    for column in df.columns:

        schema_info = schema_features.get(
            column,
            {}
        )

        series = df[column]

        is_target = (
            column == target_column
        )


        # --------------------------------------------------
        # Reuse Schema Agent evidence
        # --------------------------------------------------

        missing_percentage = schema_info.get(
            "missing_percentage",
            (
                series.isnull().sum()
                / len(series)
                * 100
                if len(series) > 0
                else 0
            )
        )

        unique_count = schema_info.get(
            "unique_count",
            int(
                series.nunique(
                    dropna=True
                )
            )
        )

        unique_percentage = schema_info.get(
            "unique_percentage",
            0
        )

        is_constant = schema_info.get(
            "is_constant",
            unique_count <= 1
        )

        is_identifier_candidate = schema_info.get(
            "is_identifier_candidate",
            False
        )

        dtype = schema_info.get(
            "pandas_dtype",
            str(series.dtype)
        )


        # --------------------------------------------------
        # Near constant detection
        # --------------------------------------------------

        non_missing = series.dropna()

        dominant_value_percentage = 0.0

        if len(non_missing) > 0:

            value_counts = (
                non_missing
                .value_counts(
                    normalize=True
                )
            )

            if len(value_counts) > 0:

                dominant_value_percentage = (
                    float(
                        value_counts.iloc[0]
                    )
                    * 100
                )


        is_near_constant = (
            dominant_value_percentage >= 95
            and not is_constant
        )


        # --------------------------------------------------
        # Correlation with target
        # --------------------------------------------------

        correlation_with_target = None

        if (
            target_column is not None
            and column != target_column
            and target_column in df.columns
            and pd.api.types.is_numeric_dtype(
                series
            )
            and pd.api.types.is_numeric_dtype(
                df[target_column]
            )
        ):

            valid_data = df[
                [
                    column,
                    target_column
                ]
            ].dropna()

            if len(valid_data) > 1:

                correlation = (
                    valid_data[column]
                    .corr(
                        valid_data[
                            target_column
                        ]
                    )
                )

                if pd.notna(
                    correlation
                ):

                    correlation_with_target = round(
                        float(
                            correlation
                        ),
                        4
                    )


        results.append({

            "column": column,

            "dtype": dtype,

            "is_target": is_target,

            "missing_percentage": round(
                float(
                    missing_percentage
                ),
                2
            ),

            "unique_count": int(
                unique_count
            ),

            "unique_percentage": round(
                float(
                    unique_percentage
                ),
                2
            ),

            "is_constant": bool(
                is_constant
            ),

            "is_near_constant": bool(
                is_near_constant
            ),

            "dominant_value_percentage": round(
                dominant_value_percentage,
                2
            ),

            "is_identifier_candidate": bool(
                is_identifier_candidate
            ),

            "correlation_with_target":
                correlation_with_target

        })


    # ======================================================
    # Feature-to-feature correlations
    # ======================================================

    numeric_df = df.select_dtypes(
        include=np.number
    )

    correlation_pairs = []

    if numeric_df.shape[1] >= 2:

        correlation_matrix = (
            numeric_df.corr()
        )

        columns = list(
            correlation_matrix.columns
        )

        for i in range(
            len(columns)
        ):

            for j in range(
                i + 1,
                len(columns)
            ):

                column_1 = columns[i]
                column_2 = columns[j]

                correlation = (
                    correlation_matrix.loc[
                        column_1,
                        column_2
                    ]
                )

                if pd.notna(
                    correlation
                ):

                    correlation_pairs.append({

                        "column_1": column_1,

                        "column_2": column_2,

                        "correlation": round(
                            float(
                                correlation
                            ),
                            4
                        ),

                        "absolute_correlation":
                            round(
                                abs(
                                    float(
                                        correlation
                                    )
                                ),
                                4
                            )

                    })


    correlation_pairs.sort(
        key=lambda x:
        x["absolute_correlation"],
        reverse=True
    )


    high_correlation_pairs = [

        pair

        for pair in correlation_pairs

        if pair[
            "absolute_correlation"
        ] >= 0.85

    ]


    return {

        "target_column":
            target_column,

        "features":
            results,

        "high_correlation_pairs":
            high_correlation_pairs

    }