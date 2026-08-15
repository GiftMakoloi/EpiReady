import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EpiReady",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    .epiready-title {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .epiready-subtitle {
        font-size: 1.15rem;
        color: #666666;
        margin-bottom: 2rem;
    }

    .metric-card {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 1rem;
        background: #ffffff;
    }

    .status-ready {
        color: #137333;
        font-weight: 700;
    }

    .status-warning {
        color: #9a6700;
        font-weight: 700;
    }

    .status-critical {
        color: #b42318;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .small-note {
        font-size: 0.85rem;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

@st.cache_data
def generate_synthetic_tb_data(n=2500, seed=42):

    rng = np.random.default_rng(seed)

    age = np.clip(
        rng.normal(42, 17, n),
        18,
        90
    ).round().astype(int)

    sex = rng.choice(
        ["Female", "Male"],
        size=n,
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
        size=n
    )

    hiv_probability = (
        0.15
        + 0.0025 * age
        + 0.05 * (sex == "Female")
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
        90
        - 0.25 * age
        - 10 * (smoking == "Yes")
        - 8 * (hiv == "Positive")
        + rng.normal(0, 15, n),
        5,
        100
    )

    logit = (
        -3.0
        + 0.025 * age
        + 0.85 * (hiv == "Positive")
        + 0.75 * (smoking == "Yes")
        - 0.035 * adherence
    )

    probability = 1 / (1 + np.exp(-logit))

    treatment_failure = np.where(
        rng.random(n) < probability,
        "Failure",
        "Success"
    )

    start_dates = pd.to_datetime(
        "2025-01-01"
    ) + pd.to_timedelta(
        rng.integers(0, 365, n),
        unit="D"
    )

    duration = rng.integers(
        90,
        240,
        n
    )

    end_dates = start_dates + pd.to_timedelta(
        duration,
        unit="D"
    )

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "province": province,
        "hiv_status": hiv,
        "smoking": smoking,
        "adherence_percent": adherence.round(1),
        "treatment_outcome": treatment_failure,
        "treatment_start": start_dates,
        "treatment_end": end_dates
    })

    # Introduce missingness
    hiv_missing = rng.random(n) < 0.20
    smoking_missing = rng.random(n) < 0.10
    adherence_missing = rng.random(n) < 0.04

    df.loc[hiv_missing, "hiv_status"] = np.nan
    df.loc[smoking_missing, "smoking"] = np.nan
    df.loc[adherence_missing, "adherence_percent"] = np.nan

    # Introduce a small number of invalid dates
    invalid_indices = rng.choice(
        n,
        size=max(5, int(n * 0.015)),
        replace=False
    )

    df.loc[
        invalid_indices,
        "treatment_end"
    ] = (
        df.loc[
            invalid_indices,
            "treatment_start"
        ]
        - pd.to_timedelta(
            rng.integers(1, 60, len(invalid_indices)),
            unit="D"
        )
    )

    return df


# ============================================================
# DATA PROFILING
# ============================================================

def profile_dataset(df):

    profile = pd.DataFrame({
        "Variable": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Rows": [
            len(df)
            for _ in df.columns
        ],
        "Missing": [
            df[col].isna().sum()
            for col in df.columns
        ],
        "Missing %": [
            round(
                df[col].isna().mean() * 100,
                2
            )
            for col in df.columns
        ],
        "Unique Values": [
            df[col].nunique(dropna=True)
            for col in df.columns
        ]
    })

    return profile


# ============================================================
# MISSING DATA ANALYSIS
# ============================================================

def missingness_analysis(df):

    records = []

    for col in df.columns:

        missing_count = int(
            df[col].isna().sum()
        )

        missing_pct = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        if missing_pct == 0:
            status = "Complete"
        elif missing_pct < 5:
            status = "Low"
        elif missing_pct < 20:
            status = "Moderate"
        else:
            status = "High"

        records.append({
            "Variable": col,
            "Missing Count": missing_count,
            "Missing %": round(missing_pct, 2),
            "Assessment": status
        })

    return pd.DataFrame(records)


# ============================================================
# DATE CONSISTENCY
# ============================================================

def date_consistency_check(df):

    result = {
        "checked": False,
        "invalid_count": 0,
        "message": "No treatment date variables detected."
    }

    start_candidates = [
        c for c in df.columns
        if "start" in c.lower()
    ]

    end_candidates = [
        c for c in df.columns
        if "end" in c.lower()
    ]

    if not start_candidates or not end_candidates:
        return result

    start_col = start_candidates[0]
    end_col = end_candidates[0]

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

    count = int(invalid.sum())

    result["checked"] = True
    result["invalid_count"] = count
    result["message"] = (
        f"{count} records contain an end date "
        f"before the start date."
    )

    return result


# ============================================================
# OUTCOME ASSESSMENT
# ============================================================

def assess_outcome(df, outcome):

    if outcome not in df.columns:
        return {
            "status": "Critical",
            "message": "Selected outcome does not exist."
        }

    series = df[outcome].dropna()

    if len(series) == 0:
        return {
            "status": "Critical",
            "message": "Outcome contains no usable observations."
        }

    unique = series.nunique()

    if unique != 2:
        return {
            "status": "Warning",
            "message": (
                f"Outcome contains {unique} observed categories. "
                "Binary logistic regression requires two outcome categories."
            )
        }

    counts = series.value_counts()

    smallest_pct = (
        counts.min()
        / counts.sum()
        * 100
    )

    if smallest_pct < 5:
        return {
            "status": "Warning",
            "message": (
                "One outcome category represents less than 5% "
                "of observed outcomes. Sparse outcome events may "
                "affect model stability."
            )
        }

    return {
        "status": "Ready",
        "message": "Binary outcome structure detected."
    }


# ============================================================
# NUMERICAL CORRELATION
# ============================================================

def numerical_correlations(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr()


# ============================================================
# VIF
# ============================================================

def calculate_vif(df):

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    numeric = numeric.dropna()

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(numeric) < 20:
        return pd.DataFrame()

    values = numeric.values

    results = []

    for i, column in enumerate(
        numeric.columns
    ):

        try:
            vif = variance_inflation_factor(
                values,
                i
            )
        except Exception:
            vif = np.nan

        results.append({
            "Variable": column,
            "VIF": round(vif, 3)
        })

    return pd.DataFrame(results)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def prepare_logistic_data(
    df,
    outcome,
    predictors
):

    working = df[
        [outcome] + predictors
    ].copy()

    working = working.dropna()

    y = working[outcome]

    unique = list(
        y.dropna().unique()
    )

    if len(unique) != 2:
        return None, None, None

    mapping = {
        unique[0]: 0,
        unique[1]: 1
    }

    y = y.map(mapping)

    X = working[predictors].copy()

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
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


def run_logistic_regression(
    df,
    outcome,
    predictors
):

    if len(predictors) == 0:
        return None

    X, y, mapping = prepare_logistic_data(
        df,
        outcome,
        predictors
    )

    if X is None:
        return None

    if len(y) < 50:
        return None

    if y.nunique() != 2:
        return None

    try:

        model = sm.Logit(
            y,
            X
        ).fit(
            disp=False
        )

        params = model.params
        conf = model.conf_int()

        results = pd.DataFrame({
            "Variable": params.index,
            "Coefficient": params.values,
            "Odds Ratio": np.exp(params.values),
            "CI Lower": np.exp(conf[0].values),
            "CI Upper": np.exp(conf[1].values),
            "P Value": model.pvalues.values
        })

        predictions = model.predict(X)

        auc = roc_auc_score(
            y,
            predictions
        )

        return {
            "model": model,
            "results": results,
            "n": len(y),
            "events": int(y.sum()),
            "auc": auc,
            "mapping": mapping
        }

    except Exception as exc:

        return {
            "error": str(exc)
        }


# ============================================================
# READINESS ASSESSMENT
# ============================================================

def calculate_readiness(
    df,
    outcome,
    analysis,
    date_result
):

    score = 100

    findings = []

    missing = df.isna().mean() * 100

    high_missing = missing[
        missing >= 20
    ]

    moderate_missing = missing[
        (missing >= 5)
        & (missing < 20)
    ]

    if len(high_missing) > 0:

        score -= min(
            25,
            len(high_missing) * 8
        )

        findings.append({
            "severity": "Critical",
            "domain": "Missingness",
            "finding": (
                f"{len(high_missing)} variable(s) have "
                "at least 20% missing observations."
            )
        })

    elif len(moderate_missing) > 0:

        score -= min(
            15,
            len(moderate_missing) * 3
        )

        findings.append({
            "severity": "Warning",
            "domain": "Missingness",
            "finding": (
                f"{len(moderate_missing)} variable(s) have "
                "5–20% missing observations."
            )
        })

    if date_result["invalid_count"] > 0:

        score -= 15

        findings.append({
            "severity": "Critical",
            "domain": "Temporal consistency",
            "finding": date_result["message"]
        })

    outcome_result = assess_outcome(
        df,
        outcome
    )

    if outcome_result["status"] == "Critical":

        score -= 25

        findings.append({
            "severity": "Critical",
            "domain": "Outcome",
            "finding": outcome_result["message"]
        })

    elif outcome_result["status"] == "Warning":

        score -= 10

        findings.append({
            "severity": "Warning",
            "domain": "Outcome",
            "finding": outcome_result["message"]
        })

    if analysis == "Logistic regression":

        if outcome_result["status"] == "Ready":

            findings.append({
                "severity": "Information",
                "domain": "Model suitability",
                "finding": (
                    "Binary outcome structure is compatible "
                    "with logistic regression."
                )
            })

    score = max(
        0,
        min(100, int(score))
    )

    if score >= 80:
        status = "Ready"
    elif score >= 60:
        status = "Proceed with caution"
    else:
        status = "Not ready"

    return score, status, findings


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    df,
    research_question,
    analysis,
    outcome,
    score,
    status,
    findings
):

    lines = []

    lines.append(
        "EpiReady Analysis Readiness Report"
    )

    lines.append("=" * 50)
    lines.append("")

    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    lines.append("")

    lines.append(
        "Research question:"
    )

    lines.append(
        research_question
    )

    lines.append("")

    lines.append(
        f"Planned analysis: {analysis}"
    )

    lines.append(
        f"Outcome: {outcome}"
    )

    lines.append("")

    lines.append(
        f"Analysis readiness score: {score}/100"
    )

    lines.append(
        f"Assessment: {status}"
    )

    lines.append("")

    lines.append(
        "Dataset:"
    )

    lines.append(
        f"Rows: {len(df)}"
    )

    lines.append(
        f"Columns: {len(df.columns)}"
    )

    lines.append("")

    lines.append(
        "Findings:"
    )

    for finding in findings:

        lines.append(
            f"[{finding['severity']}] "
            f"{finding['domain']}: "
            f"{finding['finding']}"
        )

    lines.append("")

    lines.append(
        "Interpretation:"
    )

    lines.append(
        "This assessment is an analytical decision-support "
        "prototype. It does not establish causal relationships "
        "and does not replace epidemiological or biostatistical "
        "judgment."
    )

    return "\n".join(lines)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="epiready-title">EpiReady</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="epiready-subtitle">'
    'Epidemiological Analysis Readiness Engine'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Assess whether a health dataset is suitable for the "
    "specific epidemiological question and statistical "
    "analysis you intend to perform."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Project Configuration"
)

data_source = st.sidebar.radio(
    "Data source",
    [
        "Synthetic TB dataset",
        "Upload CSV"
    ]
)

if data_source == "Synthetic TB dataset":

    n = st.sidebar.slider(
        "Synthetic sample size",
        min_value=500,
        max_value=10000,
        value=2500,
        step=500
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=1,
        value=42
    )

    df = generate_synthetic_tb_data(
        n=n,
        seed=seed
    )

else:

    uploaded = st.sidebar.file_uploader(
        "Upload CSV dataset",
        type=["csv"]
    )

    if uploaded is None:

        st.info(
            "Upload a CSV dataset or select the synthetic "
            "TB dataset from the sidebar."
        )

        st.stop()

    try:

        df = pd.read_csv(
            uploaded
        )

    except Exception as exc:

        st.error(
            f"Unable to read CSV: {exc}"
        )

        st.stop()


# ============================================================
# RESEARCH QUESTION
# ============================================================

st.sidebar.header(
    "Research Question"
)

research_question = st.sidebar.text_area(
    "Define the epidemiological question",
    value=(
        "What factors are associated with "
        "unsuccessful TB treatment?"
    )
)

analysis = st.sidebar.selectbox(
    "Planned statistical analysis",
    [
        "Logistic regression",
        "Descriptive analysis",
        "Survival analysis",
        "Diagnostic test analysis",
        "Count outcome analysis"
    ]
)


# ============================================================
# VARIABLE SELECTION
# ============================================================

st.sidebar.header(
    "Analysis Variables"
)

columns = df.columns.tolist()

default_outcome = (
    "treatment_outcome"
    if "treatment_outcome" in columns
    else columns[0]
)

outcome = st.sidebar.selectbox(
    "Outcome variable",
    columns,
    index=columns.index(
        default_outcome
    )
)

candidate_predictors = [
    c for c in columns
    if c != outcome
]

predictors = st.sidebar.multiselect(
    "Potential predictors",
    candidate_predictors,
    default=[
        c for c in [
            "age",
            "sex",
            "hiv_status",
            "smoking",
            "adherence_percent"
        ]
        if c in candidate_predictors
    ]
)


# ============================================================
# DATE CHECK
# ============================================================

date_result = date_consistency_check(
    df
)


# ============================================================
# READINESS
# ============================================================

score, status, findings = calculate_readiness(
    df,
    outcome,
    analysis,
    date_result
)


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Analysis Readiness</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Readiness Score",
        f"{score}/100"
    )

with m2:
    st.metric(
        "Observations",
        f"{len(df):,}"
    )

with m3:
    st.metric(
        "Variables",
        f"{len(df.columns)}"
    )

with m4:
    missing_total = int(
        df.isna().sum().sum()
    )

    st.metric(
        "Missing Cells",
        f"{missing_total:,}"
    )


if status == "Ready":

    st.success(
        "Assessment: Ready for the selected preliminary workflow."
    )

elif status == "Proceed with caution":

    st.warning(
        "Assessment: Proceed with caution. Important analytical issues require investigation."
    )

else:

    st.error(
        "Assessment: Not ready. Critical issues require attention before analysis."
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Data Quality",
        "Epidemiological Risks",
        "Statistical Analysis",
        "Sensitivity",
        "Report"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Research Definition</div>',
        unsafe_allow_html=True
    )

    st.write(
        research_question
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            "**Planned analysis**"
        )

        st.write(
            analysis
        )

    with c2:

        st.write(
            "**Outcome**"
        )

        st.write(
            outcome
        )

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">Dataset Profile</div>',
        unsafe_allow_html=True
    )

    profile = profile_dataset(
        df
    )

    st.dataframe(
        profile,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Missingness</div>',
        unsafe_allow_html=True
    )

    missing = missingness_analysis(
        df
    )

    st.dataframe(
        missing,
        use_container_width=True
    )

    missing_plot = missing[
        missing["Missing %"] > 0
    ]

    if not missing_plot.empty:

        fig = px.bar(
            missing_plot,
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

    st.markdown(
        '<div class="section-title">Temporal Consistency</div>',
        unsafe_allow_html=True
    )

    if date_result["checked"]:

        if date_result["invalid_count"] == 0:

            st.success(
                "No invalid treatment date sequences were detected."
            )

        else:

            st.error(
                date_result["message"]
            )

    else:

        st.info(
            date_result["message"]
        )


# ============================================================
# EPIDEMIOLOGICAL RISKS
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">Risk Findings</div>',
        unsafe_allow_html=True
    )

    if not findings:

        st.success(
            "No major automated findings were detected."
        )

    else:

        findings_df = pd.DataFrame(
            findings
        )

        st.dataframe(
            findings_df,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">Outcome Assessment</div>',
        unsafe_allow_html=True
    )

    outcome_result = assess_outcome(
        df,
        outcome
    )

    st.write(
        f"Status: {outcome_result['status']}"
    )

    st.write(
        outcome_result["message"]
    )

    if outcome in df.columns:

        counts = (
            df[outcome]
            .value_counts(dropna=False)
            .reset_index()
        )

        counts.columns = [
            "Outcome",
            "Count"
        ]

        fig = px.bar(
            counts,
            x="Outcome",
            y="Count",
            title="Outcome Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">Statistical Diagnostics</div>',
        unsafe_allow_html=True
    )

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] >= 2:

        correlation = numerical_correlations(
            df
        )

        st.write(
            "Numerical variable correlation matrix"
        )

        st.dataframe(
            correlation.round(3),
            use_container_width=True
        )

        fig = px.imshow(
            correlation,
            text_auto=True,
            title="Numerical Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "At least two numerical variables are required "
            "for correlation diagnostics."
        )

    st.markdown(
        '<div class="section-title">Variance Inflation Diagnostics</div>',
        unsafe_allow_html=True
    )

    vif = calculate_vif(
        df
    )

    if vif.empty:

        st.info(
            "Insufficient numerical data for VIF calculation."
        )

    else:

        st.dataframe(
            vif,
            use_container_width=True
        )

        high_vif = vif[
            vif["VIF"] >= 5
        ]

        if len(high_vif) > 0:

            st.warning(
                "One or more numerical variables have VIF values "
                "of at least 5. Investigate potential predictor dependence."
            )


# ============================================================
# SENSITIVITY
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">Sensitivity Analysis</div>',
        unsafe_allow_html=True
    )

    if analysis != "Logistic regression":

        st.info(
            "The current MVP sensitivity workflow is implemented "
            "for logistic regression."
        )

    elif len(predictors) == 0:

        st.warning(
            "Select at least one predictor."
        )

    else:

        result = run_logistic_regression(
            df,
            outcome,
            predictors
        )

        if result is None:

            st.error(
                "The selected variables could not be used for "
                "the logistic regression."
            )

        elif "error" in result:

            st.error(
                f"Model fitting failed: {result['error']}"
            )

        else:

            st.write(
                f"Complete-case observations: {result['n']:,}"
            )

            st.write(
                f"Observed events: {result['events']:,}"
            )

            st.write(
                f"Model AUC: {result['auc']:.3f}"
            )

            model_results = result["results"].copy()

            model_results = model_results[
                model_results["Variable"] != "const"
            ]

            st.dataframe(
                model_results.round(4),
                use_container_width=True
            )

            st.caption(
                "This MVP uses complete-case analysis for the primary model. "
                "The absence of a statistically significant result should "
                "not be interpreted as evidence of no association."
            )


# ============================================================
# REPORT
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">Analysis Readiness Report</div>',
        unsafe_allow_html=True
    )

    report = generate_report(
        df=df,
        research_question=research_question,
        analysis=analysis,
        outcome=outcome,
        score=score,
        status=status,
        findings=findings
    )

    st.text_area(
        "Report",
        report,
        height=450
    )

    st.download_button(
        label="Download Readiness Report",
        data=report,
        file_name="epiready_analysis_readiness_report.txt",
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EpiReady is a research software prototype. Automated "
    "assessments are intended to support, not replace, "
    "epidemiological and biostatistical judgment."
)

st.caption(
    "Developed by Gift Makoloi | DevOps Engineer | "
    "Software Engineer | AI Product Manager | Digital Social Scientist"
)
