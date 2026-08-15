import io
import re
import warnings
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EpiReady",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
    }

    .title {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .section {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .reason-box {
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 1rem;
        background: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SYNTHETIC TB DATA
# ============================================================

@st.cache_data
def generate_synthetic_tb_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    age = np.clip(
        rng.normal(42, 16, n),
        18,
        85
    ).round().astype(int)

    sex = rng.choice(
        ["Female", "Male"],
        n,
        p=[0.55, 0.45]
    )

    province = rng.choice(
        [
            "Gauteng",
            "KwaZulu-Natal",
            "Western Cape",
            "Eastern Cape",
            "Limpopo",
            "Mpumalanga",
            "North West",
            "Free State",
            "Northern Cape"
        ],
        n
    )

    hiv_probability = np.clip(
        0.12 + 0.0025 * age + 0.05 * (sex == "Female"),
        0.05,
        0.60
    )

    hiv = np.where(
        rng.random(n) < hiv_probability,
        "Positive",
        "Negative"
    )

    smoking = np.where(
        rng.random(n) < (0.25 + 0.002 * age),
        "Yes",
        "No"
    )

    adherence = np.clip(
        88
        - 0.20 * age
        - 9 * (smoking == "Yes")
        - 7 * (hiv == "Positive")
        + rng.normal(0, 13, n),
        5,
        100
    ).round(1)

    logit = (
        -3.0
        + 0.025 * age
        + 0.85 * (hiv == "Positive")
        + 0.70 * (smoking == "Yes")
        - 0.035 * adherence
    )

    probability = 1 / (1 + np.exp(-logit))

    outcome = np.where(
        rng.random(n) < probability,
        "Failure",
        "Success"
    )

    start_dates = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            rng.integers(0, 365, n),
            unit="D"
        )
    )

    duration = rng.integers(
        90,
        240,
        n
    )

    end_dates = (
        start_dates
        + pd.to_timedelta(
            duration,
            unit="D"
        )
    )

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "province": province,
        "hiv_status": hiv,
        "smoking": smoking,
        "adherence_percent": adherence,
        "treatment_outcome": outcome,
        "treatment_start": start_dates,
        "treatment_end": end_dates
    })

    # Deliberate missingness
    df.loc[rng.random(n) < 0.18, "hiv_status"] = np.nan
    df.loc[rng.random(n) < 0.10, "smoking"] = np.nan
    df.loc[rng.random(n) < 0.05, "adherence_percent"] = np.nan

    # Deliberate invalid dates
    bad_idx = rng.choice(
        n,
        size=15,
        replace=False
    )

    df.loc[
        bad_idx,
        "treatment_end"
    ] = (
        pd.to_datetime(df.loc[bad_idx, "treatment_start"])
        - pd.to_timedelta(
            rng.integers(1, 45, len(bad_idx)),
            unit="D"
        )
    )

    return df


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9_%\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


# ============================================================
# RESEARCH QUESTION ENGINE
# ============================================================

QUESTION_PATTERNS = {
    "Association": [
        r"\bassociat",
        r"\brisk factor",
        r"\bdetermin",
        r"\bpredict",
        r"\bfactors?\b.*\b(outcome|disease|failure|success)\b",
        r"\brelated to\b",
    ],
    "Group comparison": [
        r"\bdiffer",
        r"\bcompare",
        r"\bcomparison",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\bbetween\b.*\bgroups?\b",
    ],
    "Survival / time-to-event": [
        r"\bsurvival\b",
        r"\btime to\b",
        r"\btime-to-event\b",
        r"\bhazard\b",
        r"\btime until\b",
        r"\bdeath\b.*\btime\b",
    ],
    "Diagnostic accuracy": [
        r"\bsensitivity\b",
        r"\bspecificity\b",
        r"\bdiagnostic accuracy\b",
        r"\baccurate\b.*\bdiagnos",
        r"\btest\b.*\bdiagnos",
    ],
    "Prevalence / frequency": [
        r"\bprevalence\b",
        r"\bproportion\b",
        r"\bfrequency\b",
        r"\bhow common\b",
        r"\bincidence\b",
    ]
}


def classify_question(question):
    text = normalize_text(question)

    scores = {}

    for category, patterns in QUESTION_PATTERNS.items():

        score = 0

        for pattern in patterns:
            if re.search(pattern, text):
                score += 1

        scores[category] = score

    best_category = max(
        scores,
        key=scores.get
    )

    if scores[best_category] == 0:
        return {
            "type": "Unsupported/unclear",
            "confidence": "Low",
            "scores": scores
        }

    if scores[best_category] >= 2:
        confidence = "High"
    else:
        confidence = "Moderate"

    return {
        "type": best_category,
        "confidence": confidence,
        "scores": scores
    }


# ============================================================
# OUTCOME IDENTIFICATION
# ============================================================

OUTCOME_KEYWORDS = {
    "treatment_outcome": [
        "treatment",
        "treatment outcome",
        "treatment failure",
        "treatment success",
        "unsuccessful",
        "success",
        "failure"
    ],
    "mortality": [
        "mortality",
        "death",
        "died",
        "survival"
    ],
    "hiv_status": [
        "hiv",
        "hiv status",
        "positive",
        "negative"
    ],
    "disease_status": [
        "disease",
        "disease status",
        "case",
        "infected"
    ]
}


def variable_relevance_score(question, column):
    text = normalize_text(question)
    col = normalize_text(column)

    score = 0.0

    if col in text:
        score += 10

    if col.replace("_", " ") in text:
        score += 9

    for word in col.replace("_", " ").split():

        if word in text:
            score += 1

    return score


def identify_outcome(df, question):

    text = normalize_text(question)
    candidates = []

    # First: direct dataset-column matching
    for column in df.columns:

        score = variable_relevance_score(
            question,
            column
        )

        if score > 0:
            candidates.append(
                (column, score, "direct")
            )

    # Second: semantic keyword matching
    for column in df.columns:

        column_text = normalize_text(
            column
        )

        for key, keywords in OUTCOME_KEYWORDS.items():

            if key == column:

                for keyword in keywords:

                    if keyword in text:
                        candidates.append(
                            (column, 6, "semantic")
                        )

    # Third: special TB inference
    if "treatment" in text:

        for column in df.columns:

            normalized_col = normalize_text(
                column
            )

            if "treatment" in normalized_col:

                candidates.append(
                    (
                        column,
                        8,
                        "treatment-context"
                    )
                )

    if not candidates:

        return {
            "variable": None,
            "confidence": "Low",
            "reason": (
                "No dataset variable could be reliably "
                "linked to the outcome language in the question."
            )
        }

    # Aggregate by variable
    aggregated = {}

    for variable, score, source in candidates:

        if variable not in aggregated:
            aggregated[variable] = {
                "score": 0,
                "sources": []
            }

        aggregated[variable]["score"] += score
        aggregated[variable]["sources"].append(
            source
        )

    ranked = sorted(
        aggregated.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    variable = ranked[0][0]
    score = ranked[0][1]["score"]

    confidence = (
        "High" if score >= 10
        else "Moderate" if score >= 5
        else "Low"
    )

    reason = (
        f"`{variable}` was selected because its name and/or "
        "related outcome terms match the wording of the "
        "research question."
    )

    return {
        "variable": variable,
        "confidence": confidence,
        "reason": reason
    }


# ============================================================
# CANDIDATE PREDICTOR IDENTIFICATION
# ============================================================

def identify_predictors(df, outcome):

    predictors = []

    for column in df.columns:

        if column == outcome:
            continue

        normalized = normalize_text(
            column
        )

        if any(
            token in normalized
            for token in [
                "id",
                "patient_id",
                "record_id",
                "date",
                "time",
                "outcome"
            ]
        ):
            continue

        if df[column].nunique(
            dropna=True
        ) <= 1:
            continue

        predictors.append(column)

    return predictors


# ============================================================
# METHOD RECOMMENDATION
# ============================================================

def recommend_analysis(
    question_type,
    df,
    outcome
):

    if outcome is None:

        return {
            "analysis": "Unable to recommend",
            "reason": (
                "An outcome variable could not be identified "
                "with sufficient confidence."
            )
        }

    if question_type == "Association":

        unique_count = df[
            outcome
        ].dropna().nunique()

        if unique_count == 2:

            return {
                "analysis": "Logistic regression",
                "reason": (
                    f"The question asks about association and "
                    f"`{outcome}` has two observed outcome categories. "
                    "Binary logistic regression is therefore a "
                    "candidate primary method."
                )
            }

        if unique_count > 2:

            return {
                "analysis": "Multinomial or ordinal model",
                "reason": (
                    f"`{outcome}` has {unique_count} observed "
                    "categories. A binary logistic model would "
                    "not directly match this outcome structure."
                )
            }

    if question_type == "Group comparison":

        return {
            "analysis": "Group comparison",
            "reason": (
                "The question appears to compare outcomes "
                "between groups. The exact statistical test "
                "depends on the outcome type and distribution."
            )
        }

    if question_type == "Survival / time-to-event":

        return {
            "analysis": "Survival analysis / Cox regression",
            "reason": (
                "The question contains time-to-event language. "
                "A survival-analysis framework is a candidate "
                "because it can account for follow-up and censoring."
            )
        }

    if question_type == "Diagnostic accuracy":

        return {
            "analysis": "Diagnostic accuracy analysis",
            "reason": (
                "The question concerns diagnostic performance. "
                "Sensitivity, specificity and related 2x2 measures "
                "are appropriate candidate analyses."
            )
        }

    if question_type == "Prevalence / frequency":

        return {
            "analysis": "Descriptive / prevalence analysis",
            "reason": (
                "The question asks about frequency or prevalence. "
                "A descriptive epidemiological analysis is the "
                "appropriate starting point."
            )
        }

    return {
        "analysis": "Manual review required",
        "reason": (
            "The question could not be mapped to a supported "
            "analysis type with sufficient confidence."
        )
    }


# ============================================================
# DATA QUALITY
# ============================================================

def profile_dataset(df):

    rows = []

    for column in df.columns:

        rows.append({
            "Variable": column,
            "Data type": str(df[column].dtype),
            "Missing": int(df[column].isna().sum()),
            "Missing %": round(
                df[column].isna().mean() * 100,
                2
            ),
            "Unique values": int(
                df[column].nunique(
                    dropna=True
                )
            )
        })

    return pd.DataFrame(rows)


def date_consistency_check(df):

    start_cols = [
        c for c in df.columns
        if "start" in normalize_text(c)
    ]

    end_cols = [
        c for c in df.columns
        if "end" in normalize_text(c)
    ]

    if not start_cols or not end_cols:

        return {
            "checked": False,
            "invalid": 0,
            "start": None,
            "end": None
        }

    start_col = start_cols[0]
    end_col = end_cols[0]

    start = pd.to_datetime(
        df[start_col],
        errors="coerce"
    )

    end = pd.to_datetime(
        df[end_col],
        errors="coerce"
    )

    invalid = (
        start.notna()
        & end.notna()
        & (end < start)
    )

    return {
        "checked": True,
        "invalid": int(invalid.sum()),
        "start": start_col,
        "end": end_col
    }


# ============================================================
# MODEL PREPARATION
# ============================================================

def make_binary_target(series):

    values = series.dropna().unique()

    if len(values) != 2:
        return None, None

    values = list(values)

    # Try to put the "negative"/success-like value at 0
    negative_words = [
        "success",
        "successful",
        "no",
        "negative",
        "survived",
        "alive",
        "control",
        "absent"
    ]

    first = values[0]
    second = values[1]

    first_text = normalize_text(first)
    second_text = normalize_text(second)

    if (
        any(word in first_text for word in negative_words)
        and not any(word in second_text for word in negative_words)
    ):
        zero_value = first
        one_value = second

    elif (
        any(word in second_text for word in negative_words)
        and not any(word in first_text for word in negative_words)
    ):
        zero_value = second
        one_value = first

    else:
        zero_value = first
        one_value = second

    mapping = {
        zero_value: 0,
        one_value: 1
    }

    return series.map(mapping), mapping


def prepare_model_data(
    df,
    outcome,
    predictors,
    impute=False
):

    working = df[
        [outcome] + predictors
    ].copy()

    if impute:

        for column in predictors:

            if pd.api.types.is_numeric_dtype(
                working[column]
            ):

                working[column] = working[column].fillna(
                    working[column].median()
                )

            else:

                mode = working[column].mode(
                    dropna=True
                )

                if not mode.empty:
                    working[column] = working[column].fillna(
                        mode.iloc[0]
                    )

    else:

        working = working.dropna()

    y, mapping = make_binary_target(
        working[outcome]
    )

    if y is None:
        return None, None, None

    X = working[predictors].copy()

    categorical = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    if categorical:

        X = pd.get_dummies(
            X,
            columns=categorical,
            drop_first=True,
            dtype=float
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid]
    y = y.loc[valid]

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    return X, y, mapping


# ============================================================
# LOGISTIC MODEL
# ============================================================

def run_logistic(
    df,
    outcome,
    predictors,
    impute=False
):

    if len(predictors) == 0:
        return {
            "success": False,
            "error": "No predictors selected."
        }

    X, y, mapping = prepare_model_data(
        df,
        outcome,
        predictors,
        impute=impute
    )

    if X is None:
        return {
            "success": False,
            "error": "Outcome is not binary."
        }

    if len(y) < 30:
        return {
            "success": False,
            "error": (
                "Fewer than 30 usable observations "
                "remain for the model."
            )
        }

    if y.nunique() != 2:
        return {
            "success": False,
            "error": "The outcome must contain two categories."
        }

    try:

        model = sm.Logit(
            y,
            X
        ).fit(
            disp=False,
            maxiter=200
        )

        conf = model.conf_int()

        results = pd.DataFrame({
            "Variable": model.params.index,
            "Odds Ratio": np.exp(
                model.params.values
            ),
            "CI Lower": np.exp(
                conf[0].values
            ),
            "CI Upper": np.exp(
                conf[1].values
            ),
            "P Value": model.pvalues.values
        })

        predictions = model.predict(X)

        auc = roc_auc_score(
            y,
            predictions
        )

        return {
            "success": True,
            "model": model,
            "results": results,
            "n": len(y),
            "events": int(y.sum()),
            "auc": float(auc),
            "mapping": mapping
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }


# ============================================================
# VIF
# ============================================================

def calculate_vif(df, predictors):

    if len(predictors) < 2:
        return pd.DataFrame()

    X = df[
        predictors
    ].copy()

    categorical = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    if categorical:

        X = pd.get_dummies(
            X,
            columns=categorical,
            drop_first=True,
            dtype=float
        )

    X = X.select_dtypes(
        include=np.number
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if X.shape[1] < 2 or len(X) < 30:
        return pd.DataFrame()

    rows = []

    for i, column in enumerate(
        X.columns
    ):

        try:
            vif = variance_inflation_factor(
                X.values,
                i
            )
        except Exception:
            vif = np.nan

        rows.append({
            "Variable": column,
            "VIF": round(
                float(vif),
                3
            ) if np.isfinite(vif) else np.nan
        })

    return pd.DataFrame(rows)


# ============================================================
# READINESS ENGINE
# ============================================================

def assess_readiness(
    df,
    outcome,
    analysis,
    predictors,
    date_info
):

    score = 100
    findings = []

    # Outcome
    if outcome is None:
        score -= 30

        findings.append({
            "Severity": "Critical",
            "Domain": "Outcome",
            "Finding": (
                "No outcome variable was reliably identified."
            )
        })

    else:

        unique = df[
            outcome
        ].dropna().nunique()

        if unique == 2:

            counts = df[
                outcome
            ].value_counts(
                dropna=True
            )

            smallest = (
                counts.min()
                / counts.sum()
                * 100
            )

            if smallest < 5:

                score -= 10

                findings.append({
                    "Severity": "Warning",
                    "Domain": "Outcome",
                    "Finding": (
                        "The smallest outcome category is below "
                        "5% of observed outcomes."
                    )
                })

            else:

                findings.append({
                    "Severity": "Pass",
                    "Domain": "Outcome",
                    "Finding": (
                        "Binary outcome detected with both "
                        "categories represented."
                    )
                })

        else:

            score -= 20

            findings.append({
                "Severity": "Warning",
                "Domain": "Outcome",
                "Finding": (
                    f"Outcome contains {unique} observed categories."
                )
            })

    # Missingness
    for column in df.columns:

        missing_pct = (
            df[column].isna().mean()
            * 100
        )

        if missing_pct >= 20:

            score -= 5

            findings.append({
                "Severity": "Critical",
                "Domain": "Missingness",
                "Finding": (
                    f"{column} has {missing_pct:.1f}% missing values."
                )
            })

        elif missing_pct >= 5:

            score -= 2

            findings.append({
                "Severity": "Warning",
                "Domain": "Missingness",
                "Finding": (
                    f"{column} has {missing_pct:.1f}% missing values."
                )
            })

    # Dates
    if date_info["checked"]:

        if date_info["invalid"] > 0:

            score -= 10

            findings.append({
                "Severity": "Critical",
                "Domain": "Temporal consistency",
                "Finding": (
                    f"{date_info['invalid']} records have an "
                    "end date earlier than the start date."
                )
            })

        else:

            findings.append({
                "Severity": "Pass",
                "Domain": "Temporal consistency",
                "Finding": (
                    "No invalid start/end date sequences detected."
                )
            })

    # Predictors
    if not predictors:

        score -= 15

        findings.append({
            "Severity": "Critical",
            "Domain": "Predictors",
            "Finding": (
                "No candidate predictor variables are available."
            )
        })

    # VIF
    vif = calculate_vif(
        df,
        predictors
    )

    if not vif.empty:

        high_vif = vif[
            vif["VIF"] >= 5
        ]

        if not high_vif.empty:

            score -= 8

            findings.append({
                "Severity": "Warning",
                "Domain": "Predictor dependence",
                "Finding": (
                    "At least one predictor has VIF >= 5. "
                    "Review potential multicollinearity."
                )
            })

        else:

            findings.append({
                "Severity": "Pass",
                "Domain": "Predictor dependence",
                "Finding": (
                    "No predictor with VIF >= 5 was detected "
                    "in the available numerical representation."
                )
            })

    score = max(
        0,
        min(
            100,
            int(score)
        )
    )

    if score >= 80:
        status = "Ready for preliminary analysis"
    elif score >= 60:
        status = "Proceed with caution"
    else:
        status = "Not analysis-ready"

    return score, status, pd.DataFrame(
        findings
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "EpiReady"
)

st.sidebar.caption(
    "Epidemiological Analysis Readiness Engine"
)

source = st.sidebar.radio(
    "Data source",
    [
        "Synthetic TB dataset",
        "Upload CSV"
    ]
)

if source == "Synthetic TB dataset":

    df = generate_synthetic_tb_data()

else:

    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded is None:

        st.info(
            "Upload a CSV dataset to begin."
        )

        st.stop()

    try:
        df = pd.read_csv(uploaded)

    except Exception as exc:

        st.error(
            f"Unable to read the uploaded CSV: {exc}"
        )

        st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">EpiReady</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Epidemiological Analysis Readiness Engine'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Define an epidemiological question, let EpiReady interpret "
    "the question, verify the interpretation against your dataset, "
    "and evaluate whether the proposed analysis is supported by "
    "the available data."
)


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    '<div class="section">1. Define the research question</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Research question",
    value=(
        "What factors are associated with unsuccessful "
        "TB treatment among patients receiving TB treatment?"
    ),
    height=110
)


# ============================================================
# QUESTION INTERPRETATION
# ============================================================

question_result = classify_question(
    question
)

outcome_result = identify_outcome(
    df,
    question
)

identified_outcome = outcome_result["variable"]

predictors = identify_predictors(
    df,
    identified_outcome
) if identified_outcome else []

recommendation = recommend_analysis(
    question_result["type"],
    df,
    identified_outcome
)

# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Question Interpretation",
        "Dataset Verification",
        "Analysis Readiness",
        "Statistical Analysis",
        "Sensitivity",
        "Report"
    ]
)

# ============================================================
# QUESTION INTERPRETATION TAB
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section">Question Interpretation</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Question type",
            question_result["type"]
        )

    with c2:
        st.metric(
            "Interpretation confidence",
            question_result["confidence"]
        )

    with c3:

        st.metric(
            "Detected outcome",
            identified_outcome
            if identified_outcome
            else "Not identified"
        )

    st.markdown(
        '<div class="section">What EpiReady detected</div>',
        unsafe_allow_html=True
    )

    interpretation_table = pd.DataFrame(
        [
            {
                "Component": "Research question",
                "EpiReady interpretation": question
            },
            {
                "Component": "Question type",
                "EpiReady interpretation": question_result["type"]
            },
            {
                "Component": "Outcome variable",
                "EpiReady interpretation": (
                    identified_outcome
                    if identified_outcome
                    else "Not identified"
                )
            },
            {
                "Component": "Outcome confidence",
                "EpiReady interpretation": outcome_result["confidence"]
            },
            {
                "Component": "Candidate predictors",
                "EpiReady interpretation": (
                    ", ".join(predictors)
                    if predictors
                    else "None identified"
                )
            },
            {
                "Component": "Recommended analysis",
                "EpiReady interpretation": recommendation["analysis"]
            }
        ]
    )

    st.dataframe(
        interpretation_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section">Why?</div>',
        unsafe_allow_html=True
    )

    st.info(
        outcome_result["reason"]
    )

    st.info(
        recommendation["reason"]
    )

    with st.expander(
        "Show the rule-based interpretation logic"
    ):

        st.write(
            "EpiReady does not invent a statistical interpretation "
            "using an opaque model. The prototype uses explicit "
            "pattern-matching and dataset-validation rules."
        )

        st.write(
            "Question classification rules:"
        )

        for category, patterns in QUESTION_PATTERNS.items():

            st.write(
                f"{category}: "
                + ", ".join(patterns)
            )


# ============================================================
# DATASET VERIFICATION
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section">Dataset Verification</div>',
        unsafe_allow_html=True
    )

    st.write(
        "EpiReady now verifies whether the variables it identified "
        "actually exist in the uploaded dataset."
    )

    verification_rows = []

    if identified_outcome:

        verification_rows.append({
            "Component": "Outcome",
            "Variable": identified_outcome,
            "Exists in dataset": (
                "Yes"
                if identified_outcome in df.columns
                else "No"
            ),
            "Observed categories": (
                ", ".join(
                    map(
                        str,
                        df[
                            identified_outcome
                        ]
                        .dropna()
                        .unique()
                    )
                )
                if identified_outcome in df.columns
                else "N/A"
            )
        })

    for predictor in predictors:

        verification_rows.append({
            "Component": "Predictor",
            "Variable": predictor,
            "Exists in dataset": "Yes",
            "Observed categories": (
                f"{df[predictor].nunique(dropna=True)} unique"
            )
        })

    if verification_rows:

        st.dataframe(
            pd.DataFrame(
                verification_rows
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "EpiReady could not identify variables from the question."
        )

    st.markdown(
        '<div class="section">Dataset profile</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        profile_dataset(df),
        use_container_width=True,
        hide_index=True
    )

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    missing.columns = [
        "Variable",
        "Missing %"
    ]

    fig = px.bar(
        missing,
        x="Variable",
        y="Missing %",
        title="Missingness by Variable"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# READINESS
# ============================================================

date_info = date_consistency_check(
    df
)

readiness_score, readiness_status, findings_df = assess_readiness(
    df,
    identified_outcome,
    recommendation["analysis"],
    predictors,
    date_info
)

with tabs[2]:

    st.markdown(
        '<div class="section">Analysis Readiness</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Readiness",
            f"{readiness_score}/100"
        )

    with c2:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with c3:

        st.metric(
            "Variables",
            f"{len(df.columns)}"
        )

    with c4:

        st.metric(
            "Predictors identified",
            len(predictors)
        )

    if readiness_score >= 80:

        st.success(
            readiness_status
        )

    elif readiness_score >= 60:

        st.warning(
            readiness_status
        )

    else:

        st.error(
            readiness_status
        )

    st.markdown(
        '<div class="section">Evidence</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        findings_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section">Temporal consistency</div>',
        unsafe_allow_html=True
    )

    if date_info["checked"]:

        if date_info["invalid"] == 0:

            st.success(
                "No invalid start/end date sequences detected."
            )

        else:

            st.error(
                f"{date_info['invalid']} records have an "
                "end date earlier than their start date."
            )

    else:

        st.info(
            "No start/end date pair was automatically identified."
        )


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section">Statistical Analysis</div>',
        unsafe_allow_html=True
    )

    if recommendation[
        "analysis"
    ] == "Logistic regression":

        st.write(
            "EpiReady has identified binary logistic regression "
            "as a candidate primary analysis."
        )

        st.write(
            f"Outcome: `{identified_outcome}`"
        )

        st.write(
            "Predictors:"
        )

        st.write(
            ", ".join(predictors)
        )

        if identified_outcome:

            outcome_values = (
                df[
                    identified_outcome
                ]
                .dropna()
                .value_counts()
            )

            st.markdown(
                '<div class="section">Outcome distribution</div>',
                unsafe_allow_html=True
            )

            outcome_table = outcome_values.reset_index()

            outcome_table.columns = [
                "Outcome",
                "Count"
            ]

            st.dataframe(
                outcome_table,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            '<div class="section">Complete-case model</div>',
            unsafe_allow_html=True
        )

        model_result = run_logistic(
            df,
            identified_outcome,
            predictors,
            impute=False
        )

        if not model_result["success"]:

            st.error(
                model_result["error"]
            )

        else:

            st.write(
                f"Usable observations: {model_result['n']:,}"
            )

            st.write(
                f"Events coded as 1: {model_result['events']:,}"
            )

            st.write(
                f"Model AUC: {model_result['auc']:.3f}"
            )

            result_table = model_result[
                "results"
            ].copy()

            result_table = result_table[
                result_table["Variable"] != "const"
            ]

            st.dataframe(
                result_table.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Odds ratios and confidence intervals above are "
                "calculated from the uploaded dataset using "
                "statsmodels."
            )

        st.markdown(
            '<div class="section">Multicollinearity screening</div>',
            unsafe_allow_html=True
        )

        vif = calculate_vif(
            df,
            predictors
        )

        if vif.empty:

            st.info(
                "Insufficient compatible numerical data for VIF."
            )

        else:

            st.dataframe(
                vif,
                use_container_width=True,
                hide_index=True
            )

            if (vif["VIF"] >= 5).any():

                st.warning(
                    "At least one represented predictor has VIF >= 5. "
                    "This is a diagnostic warning, not automatic evidence "
                    "that a variable should be removed."
                )

            else:

                st.success(
                    "No represented predictor has VIF >= 5."
                )

    else:

        st.info(
            f"The rule-based engine recommends: "
            f"{recommendation['analysis']}"
        )

        st.write(
            recommendation["reason"]
        )


# ============================================================
# SENSITIVITY
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section">Sensitivity Analysis</div>',
        unsafe_allow_html=True
    )

    if recommendation[
        "analysis"
    ] != "Logistic regression":

        st.info(
            "The current prototype's automated sensitivity "
            "workflow is implemented for logistic regression."
        )

    else:

        complete_case = run_logistic(
            df,
            identified_outcome,
            predictors,
            impute=False
        )

        simple_imputation = run_logistic(
            df,
            identified_outcome,
            predictors,
            impute=True
        )

        if (
            complete_case["success"]
            and simple_imputation["success"]
        ):

            cc = complete_case["results"].copy()
            si = simple_imputation["results"].copy()

            cc = cc[
                cc["Variable"] != "const"
            ][
                ["Variable", "Odds Ratio"]
            ]

            si = si[
                si["Variable"] != "const"
            ][
                ["Variable", "Odds Ratio"]
            ]

            comparison = cc.merge(
                si,
                on="Variable",
                how="outer",
                suffixes=(
                    "_CompleteCase",
                    "_SimpleImputation"
                )
            )

            comparison["Absolute difference"] = (
                comparison[
                    "Odds Ratio_SimpleImputation"
                ]
                - comparison[
                    "Odds Ratio_CompleteCase"
                ]
            ).abs()

            st.dataframe(
                comparison.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "The sensitivity model uses simple median/mode "
                "imputation and is included as a transparent prototype. "
                "It is not a substitute for a rigorously specified "
                "multiple-imputation analysis."
            )

        else:

            st.warning(
                "A sensitivity comparison could not be completed."
            )


# ============================================================
# REPORT
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section">Auditable Report</div>',
        unsafe_allow_html=True
    )

    report_lines = [
        "EPIREADY ANALYSIS READINESS REPORT",
        "=" * 60,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "RESEARCH QUESTION",
        question,
        "",
        "QUESTION TYPE",
        question_result["type"],
        f"Confidence: {question_result['confidence']}",
        "",
        "DETECTED OUTCOME",
        str(identified_outcome),
        f"Confidence: {outcome_result['confidence']}",
        "",
        "CANDIDATE PREDICTORS",
        ", ".join(predictors),
        "",
        "RECOMMENDED ANALYSIS",
        recommendation["analysis"],
        "",
        "WHY",
        recommendation["reason"],
        "",
        "READINESS",
        f"{readiness_score}/100",
        readiness_status,
        "",
        "FINDINGS"
    ]

    for _, row in findings_df.iterrows():

        report_lines.append(
            f"[{row['Severity']}] "
            f"{row['Domain']}: "
            f"{row['Finding']}"
        )

    report_lines.extend([
        "",
        "IMPORTANT LIMITATIONS",
        "This prototype uses explicit rule-based question "
        "interpretation and statistical diagnostics.",
        "The question interpreter supports a defined set of "
        "epidemiological question patterns and should not be "
        "treated as general natural-language understanding.",
        "Automated findings do not establish causal relationships.",
        "The readiness assessment is a decision-support mechanism "
        "and not a validated clinical or regulatory score.",
    ])

    report = "\n".join(
        report_lines
    )

    st.text_area(
        "Report",
        report,
        height=650
    )

    st.download_button(
        "Download EpiReady Report",
        data=report,
        file_name="epiready_analysis_readiness_report.txt",
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EpiReady is a research software prototype designed to "
    "support transparent epidemiological and biostatistical "
    "reasoning. Automated outputs require human review."
)
