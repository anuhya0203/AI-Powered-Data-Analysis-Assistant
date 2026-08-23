import streamlit as st
import pandas as pd
import hashlib

from utils import (
    detect_column_type,
    get_numeric_series,
    is_identifier_column,
    apply_recommendations,
    recommend_outlier_strategy
)
from tools.relevance_tools import analyze_feature_relevance
from tools.schema_tools import analyze_schema
from agents.gemini_missing_agent import create_missing_value_tools
from agents.recommendation_schema import MissingValueRecommendations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from agents.gemini_outlier_agent import create_outlier_tools
from agents.outlier_recommendation_schema import OutlierRecommendations
from tools.outlier_preprocessing_tools import apply_outlier_actions

from agents.gemini_schema_agent import create_schema_tools
from agents.schema_recommendation_schema import SchemaRecommendations
from tools.schema_preprocessing_tools import apply_schema_actions

from agents.gemini_encoding_agent import create_encoding_tools
from agents.encoding_recommendation_schema import EncodingRecommendations
from tools.encoding_preprocessing_tools import (apply_encoding_actions)

from agents.gemini_scaling_agent import (create_scaling_tools)
from agents.scaling_recommendation_schema import (ScalingRecommendations)
from tools.scaling_preprocessing_tools import (apply_scaling_actions)

from agents.gemini_imbalance_agent import (create_imbalance_tools)
from agents.imbalance_recommendation_schema import (ImbalanceRecommendations)
from tools.imbalance_preprocessing_tools import (apply_imbalance_action)

from agents.gemini_relevance_agent import (create_relevance_tools)
from agents.relevance_recommendation_schema import (FeatureRelevanceRecommendations)
from tools.relevance_preprocessing_tools import (apply_relevance_actions)

from llm.gemini import get_gemini

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered AutoML & Data Analysis Assistant",
    layout="wide"
)


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "missing_recommendations" not in st.session_state:
    st.session_state.missing_recommendations = None

if "missing_analysis_signature" not in st.session_state:
    st.session_state.missing_analysis_signature = None

if "schema_recommendations" not in st.session_state:
    st.session_state.schema_recommendations = None

if "schema_analysis_signature" not in st.session_state:
    st.session_state.schema_analysis_signature = None

if "encoding_recommendations" not in st.session_state:
    st.session_state.encoding_recommendations = None

if "encoding_analysis_signature" not in st.session_state:
    st.session_state.encoding_analysis_signature = None

if "scaling_recommendations" not in st.session_state:
    st.session_state.scaling_recommendations = None

if "scaling_analysis_signature" not in st.session_state:
    st.session_state.scaling_analysis_signature = None

if "imbalance_recommendations" not in st.session_state:
    st.session_state.imbalance_recommendations = None

if "imbalance_analysis_signature" not in st.session_state:
    st.session_state.imbalance_analysis_signature = None

if "schema_analysis" not in st.session_state:
    st.session_state.schema_analysis = None

if "relevance_recommendations" not in st.session_state:
    st.session_state.relevance_recommendations = None

if "relevance_analysis_signature" not in st.session_state:
    st.session_state.relevance_analysis_signature = None

if "relevance_analysis" not in st.session_state:
    st.session_state.relevance_analysis = None

# ============================================================
# TITLE
# ============================================================

st.title("AI-Powered AutoML & Data Analysis Assistant")
st.write("Welcome! Upload a dataset to get started.")


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Detect new dataset
    # --------------------------------------------------------

    if (
        "uploaded_filename" not in st.session_state
        or st.session_state.uploaded_filename != uploaded_file.name
    ):

        st.session_state.uploaded_filename = uploaded_file.name

        st.session_state.original_df = pd.read_csv(
            uploaded_file
        )

        st.session_state.df = (
            st.session_state.original_df.copy()
        )

        # Clear previous AI recommendations
        st.session_state.missing_recommendations = None
        st.session_state.missing_analysis_signature = None
        st.session_state.duplicate_recommendation = None
        st.session_state.duplicate_analysis_signature = None
        st.session_state.encoding_recommendations = None
        st.session_state.encoding_analysis_signature = None
        st.session_state.scaling_recommendations = None
        st.session_state.scaling_analysis_signature = None
        st.session_state.imbalance_recommendations = None
        st.session_state.imbalance_analysis_signature = None
        st.session_state.schema_analysis = None
        st.session_state.relevance_recommendations = None
        st.session_state.relevance_analysis_signature = None
        st.session_state.relevance_analysis = None


    # --------------------------------------------------------
    # Create working dataframe only once
    # --------------------------------------------------------

    if "original_df" not in st.session_state:

        st.session_state.original_df = pd.read_csv(
            uploaded_file
        )


    if "df" not in st.session_state:

        st.session_state.df = (
            st.session_state.original_df.copy()
        )


    original_df = st.session_state.original_df
    df = st.session_state.df


    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader("Dataset Overview")

    st.write(
        f"Number of Rows: {df.shape[0]}"
    )

    st.write(
        f"Number of Columns: {df.shape[1]}"
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.write("### Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )


    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    missing = df.isnull().sum().sum()

    st.subheader("Dataset Summary")

    duplicates = df.duplicated().sum()

    memory = (
        df.memory_usage(deep=True).sum()
        / 1024
    )

    st.write(
        f"**Total Missing Values:** {missing}"
    )

    st.write(
        f"**Duplicate Rows:** {duplicates}"
    )

    st.write(
        f"**Memory Usage:** {memory:.2f} KB"
    )


    # ========================================================
    # SCHEMA VALIDATION AGENT
    # ========================================================

    df = st.session_state.df

    st.header("Schema Validation")


    # --------------------------------------------------------
    # Initialize Schema Agent state
    # --------------------------------------------------------

    if "schema_recommendations" not in st.session_state:
        st.session_state.schema_recommendations = None

    if "schema_analysis_signature" not in st.session_state:
        st.session_state.schema_analysis_signature = None


    # --------------------------------------------------------
    # Analyze Schema with AI
    # --------------------------------------------------------

    if st.button("🤖 Analyze Schema with AI"):

        schema_signature = hashlib.md5(
            pd.util.hash_pandas_object(
                df,
                index=True
            ).values.tobytes()
        ).hexdigest()


        # ----------------------------------------------------
        # Prevent duplicate Gemini calls
        # ----------------------------------------------------

        if (
            st.session_state.schema_analysis_signature
            == schema_signature
            and
            st.session_state.schema_recommendations
            is not None
        ):

            st.info(
                "This dataset has already been analyzed. "
                "Using the existing AI recommendations."
            )

        else:

            with st.spinner(
                "AI agent is analyzing the dataset schema..."
            ):

                # --------------------------------------------
                # Create Schema tools
                # --------------------------------------------

                tools = create_schema_tools(df)


                # --------------------------------------------
                # Get Gemini
                # --------------------------------------------

                llm = get_gemini()


                # --------------------------------------------
                # Create Schema Agent
                # --------------------------------------------

                agent = create_agent(

                    model=llm,

                    tools=tools,

                    response_format=ToolStrategy(
                        SchemaRecommendations
                    ),

                    system_prompt="""
                    You are an intelligent data preprocessing
                    agent specializing in schema validation and
                    dataset-level duplicate analysis.

                    First use the available schema analysis tool.

                    The tool returns both:
                    1. Per-column schema evidence.
                    2. Deterministic duplicate-row statistics.

                    PART 1: SCHEMA ANALYSIS

                    For every column consider:

                    - Current pandas data type
                    - Missing values
                    - Unique values
                    - Unique percentage
                    - Numeric conversion potential
                    - Datetime conversion potential
                    - Whether the column is categorical
                    - Whether the column is an identifier
                    - Whether the column is empty
                    - Whether the column is constant

                    Choose EXACTLY ONE action for every column.

                    Valid column actions:

                    No Action
                    Convert to Numeric
                    Convert to Datetime
                    Convert to Categorical
                    Drop Column
                    Treat as Identifier

                    Important rules:

                    Do not convert a column to numeric merely
                    because every value happens to be numeric.

                    Consider the semantic role of the column.

                    Do not convert identifier columns into
                    numerical features.

                    Empty columns can reasonably be dropped.

                    Constant columns may be candidates for removal,
                    but consider their semantic role.

                    Convert to datetime only when there is strong
                    evidence that the values represent dates or times.

                    Convert to categorical when the column represents
                    discrete categories rather than continuous values.

                    PART 2: DUPLICATE ANALYSIS

                    Use the duplicate statistics returned by the tool.

                    Choose EXACTLY ONE duplicate action:

                    Remove Duplicates
                    Keep Duplicates

                    Recommend Remove Duplicates when exact duplicate
                    rows appear to be accidental repeated records.

                    Recommend Keep Duplicates when repeated rows may
                    represent legitimate repeated observations or when
                    there is insufficient evidence to safely remove them.

                    If duplicate_count is zero, choose Keep Duplicates
                    and explain that no duplicate rows were detected.

                    Do not treat duplicate detection as a per-column
                    schema action.

                    Do not modify the dataset.

                    Confidence rules:

                    High:
                    Strong evidence and clear column semantics, or
                    clear duplicate evidence.

                    Medium:
                    Reasonable evidence but some uncertainty.

                    Low:
                    Ambiguous or insufficient evidence.

                    Be conservative with confidence.

                    Provide a concise reason for every column
                    recommendation and the duplicate recommendation.
                    """
                )


                # --------------------------------------------
                # Run ONE Gemini call
                # --------------------------------------------
                schema_analysis = analyze_schema(df)
                st.session_state.schema_analysis = schema_analysis
                result = agent.invoke({

                    "messages": [

                        {
                            "role": "user",

                            "content": (
                                "Analyze the schema of this dataset "
                                "and recommend the appropriate "
                                "schema action for every column."
                            )
                        }

                    ]

                })


                # --------------------------------------------
                # Store structured recommendation
                # --------------------------------------------

                st.session_state.schema_recommendations = (
                    result["structured_response"]
                )

                st.session_state.schema_analysis_signature = (
                    schema_signature
                )

                st.success(
                    "AI schema analysis completed."
                )


    # --------------------------------------------------------
    # Display Schema Analysis
    # --------------------------------------------------------

    schema_recommendations = (
        st.session_state.schema_recommendations
    )


    if schema_recommendations is not None:

        current_signature = hashlib.md5(
            pd.util.hash_pandas_object(
                df,
                index=True
            ).values.tobytes()
        ).hexdigest()


        if (
            st.session_state.schema_analysis_signature
            == current_signature
        ):

            st.subheader(
                "Schema Analysis & Recommendations"
            )


            # ------------------------------------------------
            # Create lookup for AI recommendations
            # ------------------------------------------------

            recommendation_lookup = {

                rec.column: rec

                for rec in (
                    schema_recommendations
                    .recommendations
                )

            }

            duplicate_recommendation = (
                schema_recommendations
                .duplicate_recommendation
            )

            duplicate_count = int(
                df.duplicated().sum()
            )

            duplicate_percentage = (
                duplicate_count / len(df) * 100
                if len(df) > 0
                else 0.0
            )

            st.subheader("Duplicate Analysis")

            duplicate_cols = st.columns(
                [1.5, 1.8, 4.5, 1.5]
            )

            duplicate_cols[0].markdown(
                "**Duplicate Rows**"
            )
            duplicate_cols[1].markdown(
                "**Recommendation**"
            )
            duplicate_cols[2].markdown(
                "**Reason**"
            )
            duplicate_cols[3].markdown(
                "**Confidence**"
            )

            duplicate_cols = st.columns(
                [1.5, 1.8, 4.5, 1.5]
            )

            duplicate_cols[0].write(
                f"{duplicate_count} "
                f"({duplicate_percentage:.2f}%)"
            )

            duplicate_cols[1].write(
                duplicate_recommendation.action
            )

            duplicate_cols[2].write(
                duplicate_recommendation.reason
            )

            duplicate_cols[3].write(
                duplicate_recommendation.confidence
            )

            duplicate_actions = [
                "Remove Duplicates",
                "Keep Duplicates"
            ]

            duplicate_default_index = (
                duplicate_actions.index(
                    duplicate_recommendation.action
                )
            )

            selected_duplicate_action = st.selectbox(
                "Duplicate Action",
                duplicate_actions,
                index=duplicate_default_index,
                key="schema_duplicate_action_selector"
            )


            # ------------------------------------------------
            # Analyze each column using Python
            # ------------------------------------------------

            schema_rows = []


            for column in df.columns:

                series = df[column]

                total = len(series)

                non_missing = series.dropna()

                missing_count = int(
                    series.isnull().sum()
                )

                missing_percentage = (
                    missing_count / total * 100
                    if total > 0
                    else 0
                )

                unique_count = int(
                    series.nunique(
                        dropna=True
                    )
                )

                unique_percentage = (
                    unique_count / len(non_missing) * 100
                    if len(non_missing) > 0
                    else 0
                )


                # --------------------------------------------
                # Numeric conversion test
                # --------------------------------------------

                numeric_converted = pd.to_numeric(
                    series,
                    errors="coerce"
                )

                numeric_success = (
                    numeric_converted.notna().sum()
                )

                numeric_percentage = (
                    numeric_success
                    / len(non_missing)
                    * 100
                    if len(non_missing) > 0
                    else 0
                )


                # --------------------------------------------
                # Datetime conversion test
                # --------------------------------------------

                # Never treat already-numeric columns as
                # datetime candidates simply because pandas
                # can technically convert integers to timestamps.

                if pd.api.types.is_numeric_dtype(series):

                    datetime_percentage = 0.0

                else:

                    try:

                        datetime_converted = pd.to_datetime(
                            series,
                            errors="coerce"
                        )

                        datetime_success = (
                            datetime_converted.notna().sum()
                        )

                        datetime_percentage = (
                            datetime_success
                            / len(non_missing)
                            * 100
                            if len(non_missing) > 0
                            else 0
                        )

                    except Exception:

                        datetime_percentage = 0.0


                # --------------------------------------------
                # Empty / constant / identifier detection
                # --------------------------------------------

                is_empty = (
                    len(non_missing) == 0
                )

                is_constant = (
                    unique_count <= 1
                    and not is_empty
                )

                is_identifier = (
                    unique_percentage >= 95
                    and total > 0
                )


                # --------------------------------------------
                # AI recommendation
                # --------------------------------------------

                rec = recommendation_lookup.get(
                    column
                )


                if rec is not None:

                    action = rec.action

                    reason = rec.reason

                    confidence = rec.confidence

                else:

                    action = "No Action"

                    reason = (
                        "No AI recommendation returned."
                    )

                    confidence = "Low"


                schema_rows.append({

                    "Column": column,

                    "Current Type": str(
                        series.dtype
                    ),

                    "Missing %": round(
                        missing_percentage,
                        2
                    ),

                    "Unique %": round(
                        unique_percentage,
                        2
                    ),

                    "Numeric Convert %": round(
                        numeric_percentage,
                        2
                    ),

                    "Datetime Convert %": round(
                        datetime_percentage,
                        2
                    ),

                    "Identifier?": (
                        "Yes"
                        if is_identifier
                        else "No"
                    ),

                    "Constant?": (
                        "Yes"
                        if is_constant
                        else "No"
                    ),

                    "Recommendation": action,

                    "Confidence": confidence,

                    "Reason": reason

                })


            schema_df = pd.DataFrame(
                schema_rows
            )


            # ------------------------------------------------
            # Display analysis table
            # ------------------------------------------------

            st.dataframe(
                schema_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # User Action Selection
            # ------------------------------------------------

            st.subheader(
                "Schema Actions"
            )


            selected_schema_actions = {}


            for index, row in schema_df.iterrows():

                column = row["Column"]

                recommendation = (
                    row["Recommendation"]
                )


                options = [

                    "No Action",

                    "Convert to Numeric",

                    "Convert to Datetime",

                    "Convert to Categorical",

                    "Drop Column",

                    "Treat as Identifier"

                ]


                if recommendation not in options:

                    options.insert(
                        0,
                        recommendation
                    )


                default_index = (
                    options.index(
                        recommendation
                    )
                )


                cols = st.columns(
                    [
                        1.7,
                        2.5,
                        1.5,
                        1.5,
                        1.5
                    ]
                )


                cols[0].write(
                    f"**{column}**"
                )


                cols[1].write(
                    row["Reason"]
                )


                cols[2].write(
                    f"Confidence: "
                    f"{row['Confidence']}"
                )


                cols[3].write(
                    f"AI: "
                    f"{recommendation}"
                )


                selected_schema_actions[column] = (
                    cols[4].selectbox(

                        "Action",

                        options,

                        index=default_index,

                        key=(
                            f"schema_action_"
                            f"{column}_{index}"
                        ),

                        label_visibility="collapsed"

                    )
                )


            # ------------------------------------------------
            # Apply Schema Actions
            # ------------------------------------------------

            st.subheader(
                "Apply Schema Actions"
            )


            if st.button(
                "Apply Schema Actions"
            ):

                cleaned_df, schema_summary = (
                    apply_schema_actions(
                        df,
                        selected_schema_actions,
                        selected_duplicate_action
                    )
                )


                # --------------------------------------------
                # Update dataframe
                # --------------------------------------------

                st.session_state.df = (
                    cleaned_df
                )


                # --------------------------------------------
                # Clear old AI analysis
                # --------------------------------------------

                st.session_state.schema_recommendations = None

                st.session_state.schema_analysis_signature = None


                # --------------------------------------------
                # Display result
                # --------------------------------------------

                st.success(
                    "Schema and duplicate preprocessing completed successfully!"
                )


                st.subheader(
                    "Schema Cleaning Summary"
                )


                summary_df = pd.DataFrame(
                    schema_summary
                )


                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )

    # ========================================================
    # CATEGORICAL ENCODING AGENT
    # ========================================================

    df = st.session_state.df

    st.header("Categorical Encoding")


    # --------------------------------------------------------
    # Identify categorical columns
    # --------------------------------------------------------

    categorical_columns = []

    for column in df.columns:

        series = df[column]

        # --------------------------------------------
        # Clearly categorical dtypes
        # --------------------------------------------

        if (
            pd.api.types.is_object_dtype(series)
            or
            pd.api.types.is_string_dtype(series)
            or
            isinstance(
                series.dtype,
                pd.CategoricalDtype
            )
            or
            pd.api.types.is_bool_dtype(series)
        ):

            categorical_columns.append(column)

            continue


        # --------------------------------------------
        # Detect low-cardinality categorical columns
        # that may have been stored as another dtype.
        # --------------------------------------------

        if not pd.api.types.is_numeric_dtype(series):

            non_missing = series.dropna()

            if len(non_missing) > 0:

                unique_count = (
                    non_missing.nunique()
                )

                unique_percentage = (
                    unique_count
                    / len(non_missing)
                    * 100
                )

                if (
                    unique_count <= 20
                    and
                    unique_percentage <= 20
                ):

                    categorical_columns.append(
                        column
                    )


    if not categorical_columns:

        st.info(
            "No categorical columns detected."
        )

    else:

        # ----------------------------------------------------
        # Target column selection
        # ----------------------------------------------------

        target_column = st.selectbox(

            "Select target column "
            "(required for Target Encoding)",

            ["None"] + list(df.columns),

            key="encoding_target_column"

        )


        if target_column == "None":

            target_column = None


        # ----------------------------------------------------
        # Analyze categorical features
        # ----------------------------------------------------

        if st.button(
            "🤖 Analyze Encoding with AI"
        ):

            encoding_signature = hashlib.md5(

                (
                    pd.util.hash_pandas_object(
                        df,
                        index=True
                    ).values.tobytes()
                    +
                    str(
                        target_column
                    ).encode()
                )

            ).hexdigest()


            if (

                st.session_state
                .encoding_analysis_signature
                == encoding_signature

                and

                st.session_state
                .encoding_recommendations
                is not None

            ):

                st.info(
                    "This dataset has already been analyzed "
                    "for encoding."
                )


            else:

                with st.spinner(
                    "AI agent is analyzing categorical features..."
                ):

                    # ----------------------------------------
                    # Create tools
                    # ----------------------------------------

                    tools = create_encoding_tools(

                        df,

                        target_column

                    )


                    # ----------------------------------------
                    # Gemini
                    # ----------------------------------------

                    llm = get_gemini()


                    # ----------------------------------------
                    # Create agent
                    # ----------------------------------------

                    agent = create_agent(

                        model=llm,

                        tools=tools,

                        response_format=ToolStrategy(
                            EncodingRecommendations
                        ),

                        system_prompt="""

                        You are an intelligent ML preprocessing
                        agent specializing in categorical encoding.

                        First use the available categorical analysis
                        tool.

                        For every categorical feature, consider:

                        - Number of unique categories
                        - Cardinality
                        - Unique percentage
                        - Category frequency distribution
                        - Sample category values
                        - Whether the categories appear nominal
                        - Whether the categories appear ordered
                        - Whether the feature is suitable for
                        one-hot encoding
                        - Whether the feature has high cardinality

                        Choose EXACTLY ONE action:

                        No Action
                        One-Hot Encoding
                        Frequency Encoding
                        Target Encoding
                        Ordinal Encoding

                        Decision guidelines:

                        Use One-Hot Encoding for low-cardinality
                        nominal categorical variables.

                        Use Frequency Encoding when a categorical
                        variable has relatively high cardinality
                        and one-hot encoding would create too many
                        columns.

                        Use Target Encoding when the feature has
                        high cardinality and a valid target column
                        is available.

                        Use Ordinal Encoding when the categories
                        clearly represent an ordered relationship.

                        Do NOT assume that arbitrary category names
                        have an inherent order.

                        Do not recommend Target Encoding if there
                        is no target column.

                        Do not encode the target column itself.

                        Do not modify the dataset.

                        Select exactly one action for every
                        categorical feature.

                        Confidence rules:

                        High:
                        Strong evidence for the encoding strategy.

                        Medium:
                        Reasonable evidence but some uncertainty.

                        Low:
                        Ambiguous category semantics or limited
                        evidence.

                        Give a concise reason for every decision.

                        """
                    )


                    # ----------------------------------------
                    # ONE Gemini call
                    # ----------------------------------------

                    result = agent.invoke({

                        "messages": [

                            {

                                "role": "user",

                                "content": (

                                    "Analyze the categorical "
                                    "features and recommend "
                                    "the most appropriate "
                                    "encoding strategy for "
                                    "each feature."

                                )

                            }

                        ]

                    })


                    # ----------------------------------------
                    # Store result
                    # ----------------------------------------

                    st.session_state.encoding_recommendations = (

                        result[
                            "structured_response"
                        ]

                    )

                    st.session_state.encoding_analysis_signature = (

                        encoding_signature

                    )

                    st.success(
                        "AI encoding analysis completed."
                    )


        # ----------------------------------------------------
        # Display recommendations
        # ----------------------------------------------------

        encoding_recommendations = (

            st.session_state
            .encoding_recommendations

        )


        if encoding_recommendations is not None:

            current_signature = hashlib.md5(

                (

                    pd.util.hash_pandas_object(
                        df,
                        index=True
                    ).values.tobytes()

                    +

                    str(
                        target_column
                    ).encode()

                )

            ).hexdigest()


            if (

                st.session_state
                .encoding_analysis_signature
                == current_signature

            ):

                st.subheader(
                    "Encoding Recommendations"
                )


                recommendation_lookup = {

                    rec.column: rec

                    for rec in (
                        encoding_recommendations
                        .recommendations
                    )

                }


                # --------------------------------------------
                # Header
                # --------------------------------------------

                header = st.columns(
                    [
                        1.5,
                        1.3,
                        1.3,
                        3.0,
                        1.2,
                        2.0
                    ]
                )


                header[0].markdown(
                    "**Column**"
                )

                header[1].markdown(
                    "**Categories**"
                )

                header[2].markdown(
                    "**Cardinality %**"
                )

                header[3].markdown(
                    "**Recommendation**"
                )

                header[4].markdown(
                    "**Confidence**"
                )

                header[5].markdown(
                    "**Action**"
                )


                selected_encoding_actions = {}


                # --------------------------------------------
                # Recommendation rows
                # --------------------------------------------

                for index, column in enumerate(
                    categorical_columns
                ):

                    if column == target_column:
                        continue


                    if column not in recommendation_lookup:
                        continue


                    rec = recommendation_lookup[
                        column
                    ]


                    series = df[column]

                    non_missing = series.dropna()


                    unique_count = (
                        series.nunique(
                            dropna=True
                        )
                    )


                    unique_percentage = (

                        unique_count
                        / len(non_missing)
                        * 100

                        if len(non_missing) > 0

                        else 0

                    )


                    options = [

                        "No Action",

                        "One-Hot Encoding",

                        "Frequency Encoding",

                        "Target Encoding",

                        "Ordinal Encoding"

                    ]


                    default_index = (

                        options.index(
                            rec.action
                        )

                    )


                    cols = st.columns(
                        [
                            1.5,
                            1.3,
                            1.3,
                            3.0,
                            1.2,
                            2.0
                        ]
                    )


                    cols[0].write(
                        column
                    )


                    cols[1].write(
                        unique_count
                    )


                    cols[2].write(
                        f"{unique_percentage:.2f}%"
                    )


                    cols[3].write(
                        rec.reason
                    )


                    cols[4].write(
                        rec.confidence
                    )


                    selected_encoding_actions[
                        column
                    ] = cols[5].selectbox(

                        "Action",

                        options,

                        index=default_index,

                        key=(
                            f"encoding_action_"
                            f"{column}_{index}"
                        ),

                        label_visibility="collapsed"

                    )


                # --------------------------------------------
                # Apply encoding
                # --------------------------------------------

                st.subheader(
                    "Apply Encoding"
                )


                if st.button(
                    "Apply Encoding Actions"
                ):

                    # ----------------------------------------
                    # Safety check
                    # ----------------------------------------

                    if (

                        any(
                            action
                            == "Target Encoding"

                            for action
                            in selected_encoding_actions.values()
                        )

                        and

                        target_column is None

                    ):

                        st.error(
                            "Target Encoding requires a "
                            "target column."
                        )

                    else:

                        cleaned_df, encoding_summary = (

                            apply_encoding_actions(

                                df,

                                selected_encoding_actions,

                                target_column

                            )

                        )


                        # ------------------------------------
                        # Update dataframe
                        # ------------------------------------

                        st.session_state.df = (
                            cleaned_df
                        )


                        # ------------------------------------
                        # Clear old analysis
                        # ------------------------------------

                        st.session_state.encoding_recommendations = None

                        st.session_state.encoding_analysis_signature = None


                        # ------------------------------------
                        # Display summary
                        # ------------------------------------

                        st.success(
                            "Categorical encoding completed successfully!"
                        )


                        st.subheader(
                            "Encoding Summary"
                        )


                        summary_df = pd.DataFrame(
                            encoding_summary
                        )


                        st.dataframe(
                            summary_df,

                            use_container_width=True,

                            hide_index=True

                        )

    # ========================================================
    # FEATURE RELEVANCE AGENT
    # ========================================================

    df = st.session_state.df

    st.header("Feature Relevance")


    # --------------------------------------------------------
    # Target selection
    # --------------------------------------------------------

    target_column = st.selectbox(
        "Select target column",
        ["None"] + list(df.columns),
        key="relevance_target_column"
    )

    if target_column == "None":
        target_column = None


    # --------------------------------------------------------
    # Analyze feature relevance
    # --------------------------------------------------------

    if st.button(
        "🤖 Analyze Feature Relevance with AI"
    ):

        if target_column is None:

            st.warning(
                "Please select a target column first."
            )

        elif st.session_state.schema_analysis is None:

            st.warning(
                "Please run the Schema Agent first. "
                "Feature Relevance uses its existing analysis."
            )

        else:

            relevance_signature = hashlib.md5(

                (
                    pd.util.hash_pandas_object(
                        df,
                        index=True
                    ).values.tobytes()

                    +

                    str(
                        target_column
                    ).encode()

                )

            ).hexdigest()


            # ------------------------------------------------
            # Prevent duplicate Gemini calls
            # ------------------------------------------------

            if (
                st.session_state.relevance_analysis_signature
                == relevance_signature

                and

                st.session_state.relevance_recommendations
                is not None
            ):

                st.info(
                    "This dataset has already been analyzed "
                    "for feature relevance."
                )

            else:

                with st.spinner(
                    "AI agent is analyzing feature relevance..."
                ):

                    # ----------------------------------------
                    # Reuse Schema Agent analysis
                    # ----------------------------------------

                    schema_analysis = (
                        st.session_state.schema_analysis
                    )

                    # Deterministic evidence generation.
                    # This does NOT call Gemini.
                    relevance_analysis = analyze_feature_relevance(
                        df=df,
                        target_column=target_column,
                        schema_analysis=schema_analysis
                    )

                    st.session_state.relevance_analysis = (
                        relevance_analysis
                    )

                    tools = create_relevance_tools(
                        df=df,
                        target_column=target_column,
                        schema_analysis=schema_analysis
                    )


                    llm = get_gemini()


                    agent = create_agent(

                        model=llm,

                        tools=tools,

                        response_format=ToolStrategy(
                            FeatureRelevanceRecommendations
                        ),

                        system_prompt="""

                        You are an intelligent ML Feature
                        Relevance Agent.

                        The Schema Agent has already analyzed
                        the dataset.

                        The feature relevance analysis tool
                        contains the existing Schema Agent
                        evidence.

                        DO NOT repeat schema analysis.

                        Your task is to determine whether each
                        feature should be:

                        Keep
                        Drop
                        Investigate

                        Evaluate:

                        1. Identifier-like columns
                        2. Constant columns
                        3. Near-constant columns
                        4. Excessive missingness
                        5. Highly correlated numerical features
                        6. Potential target leakage
                        7. Relationship with the target
                        8. Feature redundancy

                        IMPORTANT:

                        Never recommend dropping the target.

                        Identifier columns should generally be
                        dropped because they usually do not provide
                        meaningful predictive information.

                        Constant columns should generally be
                        dropped.

                        Near-constant columns should normally be
                        Investigated before dropping.

                        Do not drop a feature merely because its
                        correlation with the target is low.

                        Correlation measures linear numerical
                        relationships and does not capture all
                        useful relationships.

                        Highly correlated features may be redundant.
                        If the evidence is insufficient to decide
                        which feature should be removed, use
                        Investigate rather than automatically
                        dropping one.

                        Potential target leakage should be treated
                        seriously. If a feature directly contains
                        information derived from the target, recommend
                        Drop or Investigate.

                        Do not automatically drop categorical
                        features just because numerical correlation
                        is unavailable.

                        Do not modify the dataset.

                        Confidence rules:

                        High:
                        Use High only when there is strong objective evidence,
                        such as:
                        - an obvious identifier column
                        - a constant column
                        - clear target leakage
                        - the target column itself
                        - extremely strong and unambiguous redundancy

                        Medium:
                        Use Medium when the feature appears useful but the
                        evidence is not conclusive, including ordinary predictive
                        features where relevance cannot be established from the
                        available evidence alone.

                        Low:
                        Use Low when the feature's role is ambiguous or the
                        available evidence is insufficient to make a meaningful
                        judgment.

                        Do NOT assign High confidence to every feature simply
                        because the feature appears valid or has no obvious
                        problems.

                        Provide exactly one action and one concise
                        reason for every feature.

                        """

                    )


                    result = agent.invoke({

                        "messages": [

                            {

                                "role": "user",

                                "content": (

                                    "Using the existing Schema "
                                    "Agent analysis and the "
                                    "feature relevance evidence, "
                                    "analyze every feature and "
                                    "recommend whether it should "
                                    "be kept, dropped, or "
                                    "investigated."

                                )

                            }

                        ]

                    })


                    st.session_state.relevance_recommendations = (

                        result[
                            "structured_response"
                        ]

                    )


                    st.session_state.relevance_analysis_signature = (

                        relevance_signature

                    )


                    st.success(
                        "Feature relevance analysis completed."
                    )


    # --------------------------------------------------------
    # Display recommendations
    # --------------------------------------------------------

    relevance_recommendations = (
        st.session_state
        .relevance_recommendations
    )


    if relevance_recommendations is not None:
        relevance_analysis = (
        st.session_state.relevance_analysis
    )

    if st.session_state.relevance_analysis is not None:

        st.subheader(
            "Relevance Evidence"
        )

        relevance_analysis = (
            st.session_state.relevance_analysis
        )

        features = (
            relevance_analysis
            .get("features", [])
        )

        high_correlation_pairs = (
            relevance_analysis
            .get(
                "high_correlation_pairs",
                []
            )
        )

        identifier_count = sum(
            1
            for feature in features
            if feature.get(
                "is_identifier_candidate",
                False
            )
        )

        constant_count = sum(
            1
            for feature in features
            if feature.get(
                "is_constant",
                False
            )
        )

        near_constant_count = sum(
            1
            for feature in features
            if feature.get(
                "is_near_constant",
                False
            )
        )

        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Identifier-like",
            identifier_count
        )

        metric_columns[1].metric(
            "Constant",
            constant_count
        )

        metric_columns[2].metric(
            "Near-constant",
            near_constant_count
        )

        metric_columns[3].metric(
            "High Correlations",
            len(
                high_correlation_pairs
            )
        )

        metric_columns[4].metric(
            "Target",
            target_column
        )


        if high_correlation_pairs:

            st.markdown(
                "#### Highly Correlated Features"
            )

            correlation_rows = []

            for pair in high_correlation_pairs:

                correlation_rows.append({

                    "Feature 1":
                        pair["column_1"],

                    "Feature 2":
                        pair["column_2"],

                    "Correlation":
                        pair["correlation"]

                })

            st.dataframe(
                pd.DataFrame(
                    correlation_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No feature pairs with "
                "absolute correlation ≥ 0.85 "
                "were detected."
            )
        st.subheader(
            "Feature Relevance Recommendations"
        )


        selected_relevance_actions = {}


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = st.columns(
            [
                1.6,
                1.5,
                4.0,
                1.5,
                1.8
            ]
        )


        header[0].markdown(
            "**Column**"
        )

        header[1].markdown(
            "**Recommendation**"
        )

        header[2].markdown(
            "**Reason**"
        )

        header[3].markdown(
            "**Confidence**"
        )

        header[4].markdown(
            "**Action**"
        )


        # ----------------------------------------------------
        # Recommendation rows
        # ----------------------------------------------------

        for index, rec in enumerate(
            relevance_recommendations.recommendations
        ):

            if rec.column not in df.columns:
                continue


            options = [
                "Keep",
                "Drop",
                "Investigate"
            ]


            # Never allow target to be dropped
            if rec.column == target_column:

                options = [
                    "Keep"
                ]

                default_index = 0

            else:

                default_index = options.index(
                    rec.action
                )


            cols = st.columns(
                [
                    1.6,
                    1.5,
                    4.0,
                    1.5,
                    1.8
                ]
            )


            cols[0].write(
                rec.column
            )

            cols[1].write(
                rec.action
            )

            cols[2].write(
                rec.reason
            )

            cols[3].write(
                rec.confidence
            )


            selected_relevance_actions[
                rec.column
            ] = cols[4].selectbox(

                "Action",

                options,

                index=default_index,

                key=(
                    f"relevance_action_"
                    f"{rec.column}_{index}"
                ),

                label_visibility="collapsed"

            )


        # ----------------------------------------------------
        # Apply actions
        # ----------------------------------------------------

        st.subheader(
            "Apply Feature Relevance Actions"
        )


        if st.button(
            "Apply Feature Relevance Actions"
        ):

            # Safety check
            if target_column is not None:

                selected_relevance_actions[
                    target_column
                ] = "Keep"


            cleaned_df, relevance_summary = (
                apply_relevance_actions(

                    df,

                    selected_relevance_actions

                )
            )


            st.session_state.df = (
                cleaned_df
            )


            # Clear stale recommendations because
            # the dataframe has changed.

            st.session_state.relevance_recommendations = None

            st.session_state.relevance_analysis_signature = None


            st.success(
                "Feature relevance actions applied successfully!"
            )


            if relevance_summary:

                summary_df = pd.DataFrame(
                    relevance_summary
                )


                st.dataframe(

                    summary_df,

                    use_container_width=True,

                    hide_index=True

                )

    # ========================================================
    # FEATURE SCALING AGENT
    # ========================================================

    df = st.session_state.df

    st.header("Feature Scaling")


    # --------------------------------------------------------
    # Target selection
    # --------------------------------------------------------

    target_column = st.selectbox(

        "Select target column "
        "(target will not be scaled)",

        ["None"] + list(df.columns),

        key="scaling_target_column"

    )

    if target_column == "None":

        target_column = None


    # --------------------------------------------------------
    # Analyze scaling
    # --------------------------------------------------------

    if st.button(
        "🤖 Analyze Scaling with AI"
    ):

        scaling_signature = hashlib.md5(

            (
                pd.util.hash_pandas_object(
                    df,
                    index=True
                ).values.tobytes()

                +

                str(
                    target_column
                ).encode()

            )

        ).hexdigest()


        # Don't make another Gemini call
        # for the same dataset/target.

        if (

            st.session_state
            .scaling_analysis_signature
            == scaling_signature

            and

            st.session_state
            .scaling_recommendations
            is not None

        ):

            st.info(
                "This dataset has already been analyzed "
                "for feature scaling."
            )


        else:

            with st.spinner(
                "AI agent is analyzing numerical features..."
            ):

                tools = create_scaling_tools(
                    df,
                    target_column
                )

                llm = get_gemini()


                agent = create_agent(

                    model=llm,

                    tools=tools,

                    response_format=ToolStrategy(
                        ScalingRecommendations
                    ),

                    system_prompt="""

                    You are an intelligent ML preprocessing
                    agent specializing in feature scaling.

                    First use the available scaling analysis
                    tool.

                    Analyze every numerical feature and decide
                    whether scaling is appropriate.

                    Choose EXACTLY ONE action:

                    No Action
                    StandardScaler
                    MinMaxScaler
                    RobustScaler

                    Guidelines:

                    StandardScaler is appropriate for continuous
                    numerical features that are reasonably
                    symmetric and do not have substantial
                    outliers.

                    MinMaxScaler is appropriate when features
                    should be mapped to a bounded range and
                    there are no problematic extreme outliers.

                    RobustScaler is appropriate when substantial
                    outliers are present because it uses robust
                    statistics.

                    Do not scale identifier columns.

                    Do not scale categorical variables.

                    Do not scale temporal/year variables merely
                    because they are stored as integers.

                    Do not scale the target column.

                    Do not recommend scaling merely because
                    numerical values have different magnitudes.
                    Consider whether the downstream ML algorithm
                    would actually benefit from scaling.

                    For tree-based models, scaling is generally
                    unnecessary.

                    However, this system is preparing a general
                    ML-ready dataset, so recommend scaling for
                    continuous numerical features when there is
                    a clear preprocessing benefit.

                    Confidence:

                    High:
                    Strong statistical evidence.

                    Medium:
                    Reasonable evidence with some uncertainty.

                    Low:
                    Ambiguous feature role or insufficient evidence.

                    Select exactly one action for every analyzed
                    numerical feature.

                    Provide a concise reason.

                    """

                )


                result = agent.invoke({

                    "messages": [

                        {

                            "role": "user",

                            "content": (
                                "Analyze the numerical "
                                "features and recommend "
                                "the appropriate scaling "
                                "strategy for each one."
                            )

                        }

                    ]

                })


                st.session_state.scaling_recommendations = (

                    result[
                        "structured_response"
                    ]

                )

                st.session_state.scaling_analysis_signature = (

                    scaling_signature

                )

                st.success(
                    "AI scaling analysis completed."
                )


    # --------------------------------------------------------
    # Display recommendations
    # --------------------------------------------------------

    scaling_recommendations = (

        st.session_state
        .scaling_recommendations

    )


    if scaling_recommendations is not None:

        current_signature = hashlib.md5(

            (

                pd.util.hash_pandas_object(
                    df,
                    index=True
                ).values.tobytes()

                +

                str(
                    target_column
                ).encode()

            )

        ).hexdigest()


        if (

            st.session_state
            .scaling_analysis_signature
            == current_signature

        ):

            st.subheader(
                "Scaling Recommendations"
            )


            recommendation_lookup = {

                rec.column: rec

                for rec in (
                    scaling_recommendations
                    .recommendations
                )

            }


            selected_scaling_actions = {}


            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            header = st.columns(
                [
                    1.5,
                    1.3,
                    1.3,
                    1.3,
                    1.5,
                    3.0,
                    1.5,
                    2.0
                ]
            )


            header[0].markdown(
                "**Column**"
            )

            header[1].markdown(
                "**Skewness**"
            )

            header[2].markdown(
                "**Outlier %**"
            )

            header[3].markdown(
                "**Range**"
            )

            header[4].markdown(
                "**Recommendation**"
            )

            header[5].markdown(
                "**Reason**"
            )

            header[6].markdown(
                "**Confidence**"
            )

            header[7].markdown(
                "**Action**"
            )


            # ------------------------------------------------
            # Rows
            # ------------------------------------------------

            for index, rec in enumerate(
                scaling_recommendations.recommendations
            ):

                column = rec.column

                if column not in df.columns:
                    continue

                series = df[column].dropna()

                if len(series) == 0:
                    continue


                skewness = float(
                    series.skew()
                )

                q1 = series.quantile(
                    0.25
                )

                q3 = series.quantile(
                    0.75
                )

                iqr = q3 - q1


                if iqr == 0:

                    outlier_percentage = 0.0

                else:

                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                    outliers = (
                        (series < lower)
                        |
                        (series > upper)
                    )

                    outlier_percentage = (
                        outliers.sum()
                        / len(series)
                        * 100
                    )


                value_range = (
                    series.max()
                    -
                    series.min()
                )


                options = [

                    "No Action",

                    "StandardScaler",

                    "MinMaxScaler",

                    "RobustScaler"

                ]


                default_index = options.index(
                    rec.action
                )


                cols = st.columns(
                    [
                        1.5,
                        1.3,
                        1.3,
                        1.3,
                        1.5,
                        3.0,
                        1.5,
                        2.0
                    ]
                )


                cols[0].write(
                    column
                )

                cols[1].write(
                    f"{skewness:.2f}"
                )

                cols[2].write(
                    f"{outlier_percentage:.2f}%"
                )

                cols[3].write(
                    f"{value_range:.2f}"
                )

                cols[4].write(
                    rec.action
                )

                cols[5].write(
                    rec.reason
                )

                cols[6].write(
                    rec.confidence
                )


                selected_scaling_actions[
                    column
                ] = cols[7].selectbox(

                    "Action",

                    options,

                    index=default_index,

                    key=(
                        f"scaling_action_"
                        f"{column}_{index}"
                    ),

                    label_visibility="collapsed"

                )


            # ------------------------------------------------
            # Apply scaling
            # ------------------------------------------------

            st.subheader(
                "Apply Scaling"
            )


            if st.button(
                "Apply Scaling Actions"
            ):

                cleaned_df, scaling_summary = (

                    apply_scaling_actions(

                        df,

                        selected_scaling_actions

                    )

                )


                st.session_state.df = (
                    cleaned_df
                )


                st.session_state.scaling_recommendations = None

                st.session_state.scaling_analysis_signature = None


                st.success(
                    "Feature scaling completed successfully!"
                )


                st.subheader(
                    "Scaling Summary"
                )


                summary_df = pd.DataFrame(
                    scaling_summary
                )


                st.dataframe(
                    summary_df,

                    use_container_width=True,

                    hide_index=True

                )

    # ========================================================
    # CLASS IMBALANCE AGENT
    # ========================================================

    df = st.session_state.df

    st.header("Class Imbalance")


    # --------------------------------------------------------
    # Target selection
    # --------------------------------------------------------

    target_column = st.selectbox(

        "Select target column",

        ["None"] + list(df.columns),

        key="imbalance_target_column"

    )

    if target_column == "None":

        target_column = None


    # --------------------------------------------------------
    # Analyze imbalance
    # --------------------------------------------------------

    if st.button(
        "🤖 Analyze Class Imbalance with AI"
    ):

        if target_column is None:

            st.warning(
                "Please select a target column first."
            )

        else:

            imbalance_signature = hashlib.md5(

                (
                    pd.util.hash_pandas_object(
                        df,
                        index=True
                    ).values.tobytes()

                    +

                    str(
                        target_column
                    ).encode()

                )

            ).hexdigest()


            # ----------------------------------------------
            # Prevent unnecessary Gemini calls
            # ----------------------------------------------

            if (

                st.session_state
                .imbalance_analysis_signature
                == imbalance_signature

                and

                st.session_state
                .imbalance_recommendations
                is not None

            ):

                st.info(
                    "This dataset has already been analyzed "
                    "for class imbalance."
                )


            else:

                with st.spinner(
                    "AI agent is analyzing the target distribution..."
                ):

                    tools = create_imbalance_tools(
                        df,
                        target_column
                    )

                    llm = get_gemini()


                    agent = create_agent(

                        model=llm,

                        tools=tools,

                        response_format=ToolStrategy(
                            ImbalanceRecommendations
                        ),

                        system_prompt="""

                        You are an intelligent ML preprocessing
                        agent specializing in class imbalance.

                        First use the target distribution analysis
                        tool.

                        Determine whether the target represents:

                        - Classification
                        - Regression

                        Class imbalance techniques MUST NOT be
                        recommended for regression.

                        Choose exactly one action:

                        No Action
                        Class Weights
                        SMOTE
                        Random Undersampling

                        Guidelines:

                        No Action:
                        Use when the target is reasonably balanced
                        or when imbalance is not severe enough to
                        justify intervention.

                        Class Weights:
                        Prefer this as a safe general approach when
                        imbalance exists but preserving all training
                        observations is important.

                        SMOTE:
                        Consider when the minority class is
                        substantially underrepresented and there
                        are enough minority observations to create
                        synthetic samples.

                        Random Undersampling:
                        Consider when the majority class is much
                        larger than the minority class and removing
                        some majority observations is unlikely to
                        cause serious information loss.

                        Do not recommend SMOTE for regression.

                        Do not recommend any imbalance technique
                        for regression.

                        Be conservative with very small datasets.

                        If there are fewer than 5 observations in
                        the minority class, do not recommend SMOTE.

                        Consider the imbalance ratio:

                        Below 1.5:
                        Generally balanced.

                        1.5 to 3:
                        Mild imbalance.

                        3 to 5:
                        Moderate imbalance.

                        Above 5:
                        Severe imbalance.

                        Confidence:

                        High:
                        Strong evidence from class distribution.

                        Medium:
                        Reasonable evidence with some uncertainty.

                        Low:
                        Very small dataset or ambiguous target.

                        Provide one recommendation.

                        """
                    )


                    result = agent.invoke({

                        "messages": [

                            {

                                "role": "user",

                                "content": (

                                    "Analyze the target "
                                    "distribution and "
                                    "recommend whether "
                                    "class imbalance "
                                    "handling is necessary."

                                )

                            }

                        ]

                    })


                    st.session_state.imbalance_recommendations = (

                        result[
                            "structured_response"
                        ]

                    )

                    st.session_state.imbalance_analysis_signature = (

                        imbalance_signature

                    )

                    st.success(
                        "AI class imbalance analysis completed."
                    )


    # --------------------------------------------------------
    # Display recommendation
    # --------------------------------------------------------

    imbalance_recommendations = (

        st.session_state
        .imbalance_recommendations

    )


    if imbalance_recommendations is not None:

        current_signature = hashlib.md5(

            (
                pd.util.hash_pandas_object(
                    df,
                    index=True
                ).values.tobytes()

                +

                str(
                    target_column
                ).encode()

            )

        ).hexdigest()


        if (

            st.session_state
            .imbalance_analysis_signature
            == current_signature

        ):

            st.subheader(
                "Class Imbalance Recommendation"
            )


            for index, rec in enumerate(

                imbalance_recommendations
                .recommendations

            ):

                # ------------------------------------------
                # Calculate distribution for display
                # ------------------------------------------

                target_series = (
                    df[target_column]
                    .dropna()
                )

                counts = (
                    target_series
                    .value_counts()
                )


                if len(counts) > 0:

                    majority_count = int(
                        counts.iloc[0]
                    )

                    minority_count = int(
                        counts.iloc[-1]
                    )

                    imbalance_ratio = (

                        majority_count
                        / minority_count

                        if minority_count > 0
                        else 0

                    )

                else:

                    majority_count = 0
                    minority_count = 0
                    imbalance_ratio = 0


                # ------------------------------------------
                # Summary metrics
                # ------------------------------------------

                cols = st.columns(
                    [
                        1.5,
                        1.5,
                        1.5,
                        1.5
                    ]
                )


                cols[0].metric(
                    "Problem Type",
                    rec.problem_type
                )


                cols[1].metric(
                    "Classes",
                    len(counts)
                )


                cols[2].metric(
                    "Imbalance Ratio",
                    f"{imbalance_ratio:.2f}"
                )


                cols[3].metric(
                    "Minority Count",
                    minority_count
                )


                st.write(
                    f"**Target:** {rec.target_column}"
                )

                st.write(
                    f"**AI Recommendation:** "
                    f"{rec.action}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{rec.confidence}"
                )

                st.write(
                    f"**Reason:** "
                    f"{rec.reason}"
                )


                # ------------------------------------------
                # Distribution table
                # ------------------------------------------

                st.write(
                    "**Class Distribution**"
                )


                distribution_df = pd.DataFrame({

                    "Class":
                        list(counts.index),

                    "Count":
                        list(counts.values),

                    "Percentage":
                        [
                            round(
                                value
                                / len(target_series)
                                * 100,
                                2
                            )

                            for value
                            in counts.values
                        ]

                })


                st.dataframe(

                    distribution_df,

                    use_container_width=True,

                    hide_index=True

                )


                # ------------------------------------------
                # User override
                # ------------------------------------------

                options = [

                    "No Action",

                    "Class Weights",

                    "SMOTE",

                    "Random Undersampling"

                ]


                default_index = (
                    options.index(
                        rec.action
                    )
                )


                selected_action = st.selectbox(

                    "Action",

                    options,

                    index=default_index,

                    key=(
                        f"imbalance_action_"
                        f"{target_column}_{index}"
                    )

                )


                # ------------------------------------------
                # Apply
                # ------------------------------------------

                st.subheader(
                    "Apply Imbalance Strategy"
                )


                if st.button(
                    "Apply Imbalance Strategy"
                ):

                    if (
                        rec.problem_type
                        == "Regression"

                        and

                        selected_action
                        != "No Action"
                    ):

                        st.error(
                            "Imbalance handling cannot be "
                            "applied to a regression target."
                        )

                    else:

                        with st.spinner(
                            "Applying imbalance strategy..."
                        ):

                            result_df, extra_info = (
                                apply_imbalance_action(

                                    df,

                                    target_column,

                                    selected_action

                                )
                            )


                        st.session_state.df = (
                            result_df
                        )


                        st.session_state.imbalance_recommendations = None

                        st.session_state.imbalance_analysis_signature = None


                        st.success(
                            "Class imbalance preprocessing "
                            "completed successfully!"
                        )


                        # ----------------------------------
                        # Class weights
                        # ----------------------------------

                        if extra_info is not None:

                            st.subheader(
                                "Class Weights"
                            )

                            weights_df = pd.DataFrame({

                                "Class":
                                    list(
                                        extra_info.keys()
                                    ),

                                "Weight":
                                    list(
                                        extra_info.values()
                                    )

                            })

                            st.dataframe(

                                weights_df,

                                use_container_width=True,

                                hide_index=True

                            )


                        # ----------------------------------
                        # New dataset shape
                        # ----------------------------------

                        st.write(
                            f"Rows after preprocessing: "
                            f"{result_df.shape[0]}"
                        )

                        st.write(
                            f"Columns after preprocessing: "
                            f"{result_df.shape[1]}"
                        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    new_missing = df.isnull().sum().sum()

    st.write(
        f"**New Total Missing Values:** {new_missing}"
    )


    st.subheader(
        "Intelligent Missing Value Recommendations"
    )


    # --------------------------------------------------------
    # No missing values
    # --------------------------------------------------------

    if new_missing == 0:

        st.success(
            "No missing values detected in the dataset."
        )

        # Clear any old recommendations
        st.session_state.missing_recommendations = None
        st.session_state.missing_analysis_signature = None


    else:

        # ----------------------------------------------------
        # Create a signature for the current dataframe
        # ----------------------------------------------------
        #
        # This allows us to detect whether the exact same
        # dataset has already been analyzed by Gemini.
        #
        # This does NOT use the API.
        # ----------------------------------------------------

        dataframe_hash = hashlib.md5(
            pd.util.hash_pandas_object(
                df,
                index=True
            ).values.tobytes()
        ).hexdigest()


        # ----------------------------------------------------
        # Analyze Missing Values button
        # ----------------------------------------------------

        if st.button(
            "🤖 Analyze Missing Values with AI"
        ):

            # If this exact dataframe has already been
            # analyzed, do NOT call Gemini again.
            if (
                st.session_state.missing_analysis_signature
                == dataframe_hash
                and
                st.session_state.missing_recommendations
                is not None
            ):

                st.info(
                    "This dataset has already been analyzed. "
                    "Using the existing AI recommendations."
                )

            else:

                with st.spinner(
                    "AI agent is analyzing missing values..."
                ):

                    # ----------------------------------------
                    # Create tools for current dataframe
                    # ----------------------------------------

                    tools = create_missing_value_tools(
                        df
                    )


                    # ----------------------------------------
                    # Get Gemini
                    # ----------------------------------------

                    llm = get_gemini()


                    # ----------------------------------------
                    # Create LangChain agent
                    # ----------------------------------------

                    agent = create_agent(

                        model=llm,

                        tools=tools,

                        response_format=ToolStrategy(
                            MissingValueRecommendations
                        ),

                        system_prompt="""
                        You are an intelligent data
                        preprocessing agent specializing
                        in missing-value analysis.

                        Your job is to investigate the
                        dataset and recommend the most
                        appropriate preprocessing action
                        for every column containing missing
                        values.

                        Follow this process:

                        1. Profile the dataset first.

                        2. Identify ONLY columns containing
                        missing values.

                        3. For each missing-value column:
                           - Determine its data type.
                           - Analyze relevant statistics.
                           - Consider missing percentage.
                           - Consider the distribution.
                           - Consider relevant categorical
                             relationships.

                        4. If another categorical column
                        could provide useful information
                        about the missingness pattern,
                        investigate that relationship.

                        5. Do not modify the dataset.

                        6. Recommend EXACTLY ONE action for
                        every column containing missing
                        values.

                        Valid actions:

                        Mean
                        Median
                        Mode
                        Drop Column
                        Leave Missing
                        No Action

                        Never provide multiple possible
                        actions.

                        Choose the single best action based
                        on the evidence obtained from the
                        available tools.

                        Confidence rules:

                        High:
                        Strong evidence and sufficient data.

                        Medium:
                        Reasonable evidence but some
                        uncertainty.

                        Low:
                        Small sample size, weak evidence,
                        or significant uncertainty.

                        Be conservative with confidence.

                        Give a clear statistical reason for
                        every recommendation.
                        """
                    )


                    # ----------------------------------------
                    # Run ONE Gemini agent analysis
                    # ----------------------------------------

                    result = agent.invoke({

                        "messages": [

                            {
                                "role": "user",

                                "content": (
                                    "Analyze this dataset "
                                    "and determine how its "
                                    "missing values should "
                                    "be handled."
                                )
                            }

                        ]

                    })


                    # ----------------------------------------
                    # Store structured result
                    # ----------------------------------------

                    st.session_state.missing_recommendations = (
                        result["structured_response"]
                    )

                    st.session_state.missing_analysis_signature = (
                        dataframe_hash
                    )

                    st.success(
                        "AI analysis completed."
                    )


        # ----------------------------------------------------
        # Display recommendations if available
        # ----------------------------------------------------

        recommendations = (
            st.session_state.missing_recommendations
        )


        if recommendations is not None:

            # ------------------------------------------------
            # Convert Gemini output to dataframe
            # ------------------------------------------------

            recommendation_rows = []


            for rec in recommendations.recommendations:

                column = rec.column


                # Safety check:
                # Ignore columns that no longer exist.
                if column not in df.columns:
                    continue


                missing_count = int(
                    df[column].isnull().sum()
                )


                missing_percentage = (
                    df[column].isnull().mean()
                    * 100
                )


                # Calculate skewness only for numeric columns
                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    skewness = df[column].skew()

                    if pd.isna(skewness):
                        skewness = 0.0

                    skewness = round(
                        float(skewness),
                        4
                    )

                    column_type = "Numeric"

                else:

                    skewness = "N/A"
                    column_type = "Categorical"


                recommendation_rows.append({

                    "Column": column,

                    "Type": column_type,

                    "Missing Count": missing_count,

                    "Missing %": round(
                        missing_percentage,
                        2
                    ),

                    "Skewness": skewness,

                    "Recommendation": rec.action,

                    "Reason": rec.reason,

                    "Confidence": rec.confidence

                })


            recommendation_df = pd.DataFrame(
                recommendation_rows
            )


            # ------------------------------------------------
            # Store user's selected actions
            # ------------------------------------------------

            selected_actions = {}


            # ------------------------------------------------
            # Table header
            # ------------------------------------------------

            header = st.columns(
                [
                    1.5,
                    1.3,
                    1.2,
                    1,
                    0.9,
                    1.6,
                    2.8,
                    1.8
                ]
            )


            header[0].markdown("**Column**")
            header[1].markdown("**Type**")
            header[2].markdown("**Missing Count**")
            header[3].markdown("**Missing %**")
            header[4].markdown("**Skewness**")
            header[5].markdown("**Recommendation**")
            header[6].markdown("**Reason**")
            header[7].markdown("**Action**")


            # ------------------------------------------------
            # Recommendation rows
            # ------------------------------------------------

            for index, row in recommendation_df.iterrows():

                column = row["Column"]

                column_type = row["Type"]

                recommendation = (
                    row["Recommendation"]
                )


                # --------------------------------------------
                # Valid actions
                # --------------------------------------------

                if recommendation == "No Action":

                    options = [
                        "No Action",
                        "Leave Missing",
                        "Drop Column"
                    ]


                elif column_type == "Numeric":

                    options = [
                        "Mean",
                        "Median",
                        "Mode",
                        "Drop Column",
                        "Leave Missing",
                        "No Action"
                    ]


                elif column_type == "Categorical":

                    options = [
                        "Mode",
                        "Drop Column",
                        "Leave Missing",
                        "No Action"
                    ]


                else:

                    options = [
                        "Drop Column",
                        "Leave Missing",
                        "No Action"
                    ]


                # --------------------------------------------
                # Ensure AI recommendation is available
                # --------------------------------------------

                if recommendation not in options:

                    options.insert(
                        0,
                        recommendation
                    )


                default_index = (
                    options.index(
                        recommendation
                    )
                )


                # --------------------------------------------
                # Create row
                # --------------------------------------------

                cols = st.columns(
                    [
                        1.5,
                        1.3,
                        1.2,
                        1,
                        0.9,
                        1.6,
                        2.8,
                        1.8
                    ]
                )


                cols[0].write(
                    column
                )

                cols[1].write(
                    column_type
                )

                cols[2].write(
                    row["Missing Count"]
                )

                cols[3].write(
                    f'{row["Missing %"]}%'
                )

                cols[4].write(
                    row["Skewness"]
                )

                cols[5].write(
                    recommendation
                )

                cols[6].write(
                    row["Reason"]
                )


                # --------------------------------------------
                # Action dropdown
                # --------------------------------------------

                selected_actions[column] = (
                    cols[7].selectbox(

                        "Action",

                        options,

                        index=default_index,

                        key=(
                            f"missing_action_"
                            f"{column}_{index}"
                        ),

                        label_visibility="collapsed"

                    )
                )


            # ------------------------------------------------
            # Apply Recommendations
            # ------------------------------------------------

            st.subheader(
                "Apply Recommendations"
            )


            if st.button(
                "Apply Recommendations"
            ):

                before_missing = (
                    df.isnull().sum().sum()
                )


                # --------------------------------------------
                # IMPORTANT:
                # Copy AI recommendations and replace
                # Recommendation with the user's selected
                # dropdown action.
                # --------------------------------------------

                recommendations_to_apply = (
                    recommendation_df.copy()
                )


                recommendations_to_apply[
                    "Recommendation"
                ] = recommendations_to_apply[
                    "Column"
                ].map(
                    selected_actions
                )


                # --------------------------------------------
                # Apply using your existing utility
                # --------------------------------------------

                cleaned_df = apply_recommendations(
                    df,
                    recommendations_to_apply
                )


                # --------------------------------------------
                # Update working dataframe
                # --------------------------------------------

                st.session_state.df = (
                    cleaned_df
                )

                df = cleaned_df


                st.success(
                    "Recommendations applied successfully!"
                )


                # --------------------------------------------
                # Cleaning Summary
                # --------------------------------------------

                st.subheader(
                    "Cleaning Summary"
                )


                after_missing = (
                    cleaned_df.isnull().sum().sum()
                )


                st.write(
                    f"Missing Values Before: "
                    f"{before_missing}"
                )

                st.write(
                    f"Missing Values After: "
                    f"{after_missing}"
                )


                # --------------------------------------------
                # Dataset changed.
                # Previous AI recommendations are no longer
                # valid for the new dataframe.
                # --------------------------------------------

                st.session_state.missing_recommendations = None

                st.session_state.missing_analysis_signature = None


    # ========================================================
    # OUTLIER DETECTION
    # ========================================================

    df = st.session_state.df

    # ========================================================
    # OUTLIER DETECTION AGENT
    # ========================================================

    df = st.session_state.df

    st.header("Outlier Detection")


    # --------------------------------------------------------
    # Initialize Outlier Agent state
    # --------------------------------------------------------

    if "outlier_recommendations" not in st.session_state:
        st.session_state.outlier_recommendations = None

    if "outlier_analysis_columns" not in st.session_state:
        st.session_state.outlier_analysis_columns = None


    # --------------------------------------------------------
    # Get numeric columns
    # --------------------------------------------------------

    numeric_columns = []

    for column in df.columns:

        if (
            detect_column_type(df[column]) == "numeric"
            and not is_identifier_column(column)
        ):

            numeric_columns.append(column)


    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    selected_columns = st.multiselect(
        "Select columns for outlier analysis",
        numeric_columns
    )


    if selected_columns:

        # ----------------------------------------------------
        # Analyze button
        # ----------------------------------------------------

        if st.button("🤖 Analyze Selected Columns with AI"):

            # ------------------------------------------------
            # If the same columns have already been analyzed,
            # don't call Gemini again.
            # ------------------------------------------------

            if (
                st.session_state.outlier_analysis_columns
                == selected_columns
                and
                st.session_state.outlier_recommendations
                is not None
            ):

                st.info(
                    "These columns have already been analyzed. "
                    "Using the existing AI recommendations."
                )

            else:

                with st.spinner(
                    "AI agent is analyzing outliers..."
                ):

                    # ----------------------------------------
                    # Create tools
                    # ----------------------------------------

                    tools = create_outlier_tools(
                        df,
                        selected_columns
                    )


                    # ----------------------------------------
                    # Get Gemini
                    # ----------------------------------------

                    llm = get_gemini()


                    # ----------------------------------------
                    # Create agent
                    # ----------------------------------------

                    agent = create_agent(

                        model=llm,

                        tools=tools,

                        response_format=ToolStrategy(
                            OutlierRecommendations
                        ),

                        system_prompt="""
                        You are an intelligent data preprocessing
                        agent specializing in outlier detection
                        and treatment.

                        Analyze the statistical evidence provided
                        by the available tools.

                        For every selected numeric column, consider:

                        - Number of observations
                        - Number of detected outliers
                        - Outlier percentage
                        - Skewness
                        - IQR boundaries
                        - Z-score outlier count
                        - Whether the distribution appears
                        approximately symmetric or skewed

                        Choose EXACTLY ONE action for each
                        selected column.

                        Valid actions:

                        IQR
                        Winsorization
                        Z-Score
                        Remove Rows
                        No Action

                        General guidance:

                        IQR:
                        Prefer when robust outlier detection is
                        appropriate, particularly for skewed data.

                        Winsorization:
                        Prefer when extreme values should be capped
                        rather than removed.

                        Z-Score:
                        Prefer when the distribution is approximately
                        normal and standard-deviation based detection
                        is appropriate.

                        Remove Rows:
                        Use cautiously. Only recommend this when
                        removing observations is justified.

                        No Action:
                        Use when there is insufficient evidence of
                        problematic outliers.

                        Do not modify the dataset.

                        Do not give multiple possible actions.

                        Select one specific action for every column.

                        Confidence rules:

                        High:
                        Strong statistical evidence and sufficient
                        sample size.

                        Medium:
                        Reasonable evidence but some uncertainty,
                        including moderately small datasets.

                        Low:
                        Very small sample size or substantial
                        uncertainty.

                        Be conservative with confidence.

                        Provide a concise statistical reason for
                        every recommendation.
                        """
                    )


                    # ----------------------------------------
                    # Run ONE Gemini agent analysis
                    # ----------------------------------------

                    result = agent.invoke({

                        "messages": [

                            {
                                "role": "user",

                                "content": (
                                    "Analyze the selected numeric "
                                    "columns and recommend the most "
                                    "appropriate outlier treatment "
                                    "for each."
                                )
                            }

                        ]

                    })


                    # ----------------------------------------
                    # Store result
                    # ----------------------------------------

                    st.session_state.outlier_recommendations = (
                        result["structured_response"]
                    )

                    st.session_state.outlier_analysis_columns = (
                        selected_columns.copy()
                    )

                    st.success(
                        "AI outlier analysis completed."
                    )


        # ----------------------------------------------------
        # Display recommendations
        # ----------------------------------------------------

        recommendations = (
            st.session_state.outlier_recommendations
        )


        # Only display if the recommendations correspond
        # to the currently selected columns.

        if (
            recommendations is not None
            and
            st.session_state.outlier_analysis_columns
            == selected_columns
        ):

            st.subheader(
                "Outlier Recommendations"
            )


            # ------------------------------------------------
            # Convert structured recommendations to DataFrame
            # ------------------------------------------------

            recommendation_rows = []


            for rec in recommendations.recommendations:

                column = rec.column


                # Safety check
                if column not in df.columns:
                    continue


                # --------------------------------------------
                # Calculate the statistics for display
                # --------------------------------------------

                series = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).dropna()


                if len(series) > 0:

                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)

                    iqr = q3 - q1

                    lower_bound = (
                        q1 - 1.5 * iqr
                    )

                    upper_bound = (
                        q3 + 1.5 * iqr
                    )

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

                    skewness = series.skew()

                    if pd.isna(skewness):
                        skewness = 0.0

                else:

                    outlier_count = 0
                    outlier_percentage = 0.0
                    skewness = 0.0


                recommendation_rows.append({

                    "Column": column,

                    "Outliers": outlier_count,

                    "Outlier %": round(
                        float(outlier_percentage),
                        2
                    ),

                    "Skewness": round(
                        float(skewness),
                        4
                    ),

                    "Recommendation": rec.action,

                    "Reason": rec.reason,

                    "Confidence": rec.confidence

                })


            outlier_df = pd.DataFrame(
                recommendation_rows
            )


            # ------------------------------------------------
            # Store user-selected actions
            # ------------------------------------------------

            selected_outlier_actions = {}


            # ------------------------------------------------
            # Table header
            # ------------------------------------------------

            header = st.columns(
                [
                    1.6,
                    1.2,
                    1.2,
                    1.2,
                    1.6,
                    2.8,
                    1.8
                ]
            )


            header[0].markdown(
                "**Column**"
            )

            header[1].markdown(
                "**Outliers**"
            )

            header[2].markdown(
                "**Outlier %**"
            )

            header[3].markdown(
                "**Skewness**"
            )

            header[4].markdown(
                "**Recommendation**"
            )

            header[5].markdown(
                "**Reason**"
            )

            header[6].markdown(
                "**Action**"
            )


            # ------------------------------------------------
            # Recommendation rows
            # ------------------------------------------------

            for index, row in outlier_df.iterrows():

                column = row["Column"]

                recommendation = (
                    row["Recommendation"]
                )


                # --------------------------------------------
                # Valid actions
                # --------------------------------------------

                if recommendation == "No Action":

                    options = [
                        "No Action",
                        "IQR",
                        "Winsorization",
                        "Z-Score",
                        "Remove Rows"
                    ]


                elif recommendation == "IQR":

                    options = [
                        "IQR",
                        "Winsorization",
                        "Z-Score",
                        "Remove Rows",
                        "No Action"
                    ]


                elif recommendation == "Winsorization":

                    options = [
                        "Winsorization",
                        "IQR",
                        "Z-Score",
                        "Remove Rows",
                        "No Action"
                    ]


                elif recommendation == "Z-Score":

                    options = [
                        "Z-Score",
                        "IQR",
                        "Winsorization",
                        "Remove Rows",
                        "No Action"
                    ]


                elif recommendation == "Remove Rows":

                    options = [
                        "Remove Rows",
                        "IQR",
                        "Winsorization",
                        "Z-Score",
                        "No Action"
                    ]


                else:

                    options = [
                        recommendation,
                        "IQR",
                        "Winsorization",
                        "Z-Score",
                        "Remove Rows",
                        "No Action"
                    ]


                # --------------------------------------------
                # Recommended action selected by default
                # --------------------------------------------

                default_index = (
                    options.index(
                        recommendation
                    )
                )


                # --------------------------------------------
                # Create row
                # --------------------------------------------

                cols = st.columns(
                    [
                        1.6,
                        1.2,
                        1.2,
                        1.2,
                        1.6,
                        2.8,
                        1.8
                    ]
                )


                cols[0].write(
                    column
                )

                cols[1].write(
                    row["Outliers"]
                )

                cols[2].write(
                    f'{row["Outlier %"]}%'
                )

                cols[3].write(
                    row["Skewness"]
                )

                cols[4].write(
                    recommendation
                )

                cols[5].write(
                    row["Reason"]
                )


                # --------------------------------------------
                # Action dropdown
                # --------------------------------------------

                selected_outlier_actions[column] = (
                    cols[6].selectbox(

                        "Action",

                        options,

                        index=default_index,

                        key=(
                            f"outlier_action_"
                            f"{column}_{index}"
                        ),

                        label_visibility="collapsed"

                    )
                )

            st.subheader("Apply Outlier Actions")

            if st.button("Apply Outlier Actions"):

                rows_before = len(df)

                cleaned_df, outlier_summary = (
                    apply_outlier_actions(
                        df,
                        selected_outlier_actions
                    )
                )

                rows_after = len(cleaned_df)

                # Update working dataframe
                st.session_state.df = cleaned_df

                # Clear old AI recommendations because
                # the dataset has now changed.
                st.session_state.outlier_recommendations = None
                st.session_state.outlier_analysis_columns = None

                st.success(
                    "Outlier preprocessing completed successfully!"
                )

                st.subheader("Outlier Cleaning Summary")

                st.write(
                    f"**Rows Before:** {rows_before}"
                )

                st.write(
                    f"**Rows After:** {rows_after}"
                )

                st.write(
                    f"**Rows Removed:** "
                    f"{rows_before - rows_after}"
                )

                summary_df = pd.DataFrame(
                    outlier_summary
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )