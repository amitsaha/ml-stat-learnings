import numpy as np
import streamlit as st
import pandas as pd
import tarfile
from io import BytesIO

from engine import (
    CATEGORY_PREFIX,
    LABEL,
    POSITIVE_PROBABILITY_SCORE,
    calculate_metrics,
    compare_categories,
    ffh_monte_carlo_demo,
    fisher_monte_carlo_convergence,
    plot_f1_trends,
    plot_recall_trends,
)


SESSION_STATE_PERF_COMPARISON_CATEGORIES = "pairwise_category_members"
SESSION_STATE_PERF_COMPARISON_RESULT = "comparison_result"
SESSION_STATE_IMPACTED_GROUP_DATA = "impacted_group_data"


def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    # Plain CSV
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    # Single CSV packed as csv.tar.gz
    if name.endswith(".csv.tar.gz") or name.endswith(".tar.gz"):
        with tarfile.open(fileobj=uploaded_file, mode="r:gz") as tar:
            member = tar.getmembers()[0]
            csv_bytes = tar.extractfile(member).read()
            return pd.read_csv(BytesIO(csv_bytes))

    raise ValueError("Unsupported file type")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Global Significance Analysis of Classification Data", layout="wide"
)

st.title("📊 Global Significance Analysis of Classification Data")

uploaded_file = st.file_uploader(
    "Upload data file",
    type=["csv", "tar.gz"],
    help="Upload a CSV or a CSV compressed as csv.tar.gz",
)

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    data = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()


st.text("Data loaded - first 10 rows")

data.head()[["label", "score", "group:gender", "group:ethnicity", "group:age"]]


# ---- Threshold control
prob_threshold = st.slider(
    "Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

# ---- Validate required columns
required_cols = {
    LABEL,
    POSITIVE_PROBABILITY_SCORE,
}

missing = required_cols - set(data.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---- Detect subgroup categories
category_cols = [c for c in data.columns if c.startswith(CATEGORY_PREFIX)]

if not category_cols:
    st.warning("No subgroup columns found (group:*)")
    st.stop()


def reset_perf_state():
    if SESSION_STATE_PERF_COMPARISON_RESULT in st.session_state:
        del st.session_state[SESSION_STATE_PERF_COMPARISON_RESULT]
    if SESSION_STATE_PERF_COMPARISON_CATEGORIES in st.session_state:
        del st.session_state[SESSION_STATE_PERF_COMPARISON_CATEGORIES]


category = st.selectbox(
    "Select subgroup category", category_cols, on_change=reset_perf_state
)

metrics_for_analysis = ["Recall", "F1"]
metric_for_visualize = st.selectbox("Select metric to visualize", metrics_for_analysis)

st.divider()
st.subheader("Results")

# -----------------------------
# Run analysis
# -----------------------------
overall_metrics = calculate_metrics(data, prob_threshold)

grouped = data.groupby(category)

subgroup_data = dict()

for subgroup, df_sub in grouped:
    metrics = calculate_metrics(df_sub, prob_threshold)
    subgroup_data[subgroup] = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "cm_data": metrics["cm_data"],
    }


metrics = {"overall_data": overall_metrics, "subgroup_data": subgroup_data}
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Metrics",
        "Fisher exact test",
        "Fisher-Freeman-Halton Test",
        "Global Significance Analysis",
    ]
)


def run_performance_comparison(metrics_df_subset):
    return compare_categories(metrics_df_subset)


with tab1:
    # print(metrics_df.loc["subgroup_data"])
    df = pd.DataFrame.from_dict(metrics["subgroup_data"])
    # gather all tp and fn in a table
    all_tp = []
    all_fn = []
    for item in metrics["subgroup_data"].values():
        all_tp.append(item["cm_data"]["tp"])
        all_fn.append(item["cm_data"]["fn"])

    tp_fn = [(k, v) for k, v in zip(all_tp, all_fn)]

    table = np.array(tp_fn)
    df = pd.DataFrame(
        table, columns=["TP", "FN"], index=metrics["subgroup_data"].keys()
    )
    st.dataframe(df)
    # contingency_table

    if len(metrics["subgroup_data"].keys()) == 2:
        df_2x2 = df
    else:
        df_2x2 = pd.DataFrame(
            {
                "Event": [8, 2],
                "No Event": [2, 8],
            },
            index=["Group A", "Group B"],
        )

    if len(metrics["subgroup_data"].keys()) > 2:
        df_ffh = df
    else:
        df_ffh = pd.DataFrame(
            {
                "Correct": [90, 50, 10],
                "Incorrect": [10, 50, 40],
            },
            index=[
                "Group A",
                "Group B",
                "Group C",
            ],
        )

    match metric_for_visualize:
        case "Recall":
            fig = plot_recall_trends(metrics)
            st.pyplot(fig)

        case "F1":
            fig = plot_f1_trends(metrics)
            st.pyplot(fig)

with tab2:
    st.text("Fisher's exact test")

    table = df_2x2.copy()
    table["Row total"] = table.sum(axis=1)

    col_totals = table.sum(axis=0)
    table.loc["Column total"] = col_totals

    st.dataframe(table)

    from scipy.stats import fisher_exact

    odds_ratio, p_exact = fisher_exact(df_2x2.values, alternative="two-sided")

    st.metric("Fisher’s Exact p-value", f"{p_exact:.4f}")

    with st.expander("Monte carlo convergence"):
        trace_df = pd.DataFrame(fisher_monte_carlo_convergence(df_2x2.values))

        st.line_chart(trace_df.set_index("reps")["p_value"])

        st.caption(f"True Fisher p-value = {p_exact:.4f}")


with tab3:
    st.text("Demonstration of FFH test")

    st.subheader("Method")

    st.code(
        """
x = rng.hypergeometric(remaining_col[0], remaining_col[1], r) # good (TP), bad (FP), number of samples

# example call - age group
# get one cell (0, 0) for age group 1
x = rng.hypergeometric(2186, 284, 843)

# for first row
# we automatically get the next cell
[x, 843-x]

# repeat for all but the last row

        """
    )

    ROW_COLORS = {
        "Group A": "#E3F2FD",  # light blue
        "Group B": "#E8F5E9",  # light green
        "Group C": "#FFFDE7",  # light yellow
        "Group D": "#FCE4EC",  # light pink
        "Column total": "#EEEEEE",
    }

    def highlight_rows(row):
        color = ROW_COLORS.get(row.name, "#FFFFFF")
        return [f"background-color: {color}"] * len(row)

    table_with_margins = df_ffh.copy()
    table_with_margins["Row total"] = df_ffh.sum(axis=1)

    col_totals = df_ffh.sum(axis=0)
    col_totals["Row total"] = col_totals.sum()
    table_with_margins.loc["Column total"] = col_totals

    styled = table_with_margins.style.apply(highlight_rows, axis=1)
    st.dataframe(styled, width="stretch")

    _, _, simulated_tables = ffh_monte_carlo_demo(df_ffh, 5)
    for i, table in enumerate(simulated_tables):
        table = pd.DataFrame(table)
        st.subheader(f"Simulated table {i + 1}")

        table_with_margins = table.copy()
        table_with_margins["Row total"] = table.sum(axis=1)

        col_totals = table.sum(axis=0)
        col_totals["Row total"] = col_totals.sum()
        table_with_margins.loc["Column total"] = col_totals

        styled = table_with_margins.style.apply(highlight_rows, axis=1)
        st.dataframe(styled, width="stretch")

    st.title("Monte Carlo FFH: p-value convergence demo")

    rep_levels = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    rep_levels = [50_000]

    results = []

    st.dataframe(df_ffh)

    trace = fisher_monte_carlo_convergence(df_ffh.values)

    trace_df = pd.DataFrame(trace)

    st.dataframe(trace_df, width="stretch")

    st.line_chart(trace_df.set_index("reps")["p_value"])

    latest = trace_df.iloc[-1]

    st.metric(
        label="Estimated FFH p-value",
        value=f"{latest['p_value']}",
        help=(
            "Probability of observing a table at least this extreme "
            "under the null of independence, estimated via Monte Carlo."
        ),
    )

    st.write("Monte Carlo tail probability components")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Simulations run",
            int(latest["reps"]),
        )

    with col2:
        st.metric(
            "Extreme tables observed",
            latest["exceed"],
        )

    with col3:
        st.metric(
            "Estimated p-value",
            f"{latest['p_value']}",
        )


with tab4:

    def visualize_results(comparison_result):
        result = comparison_result
        st.text("Performance analysis complete")

        st.text(f"Results different globally: {result['global_significant']}")

    if SESSION_STATE_PERF_COMPARISON_RESULT in st.session_state:
        comparison_result = st.session_state[SESSION_STATE_PERF_COMPARISON_RESULT]
        visualize_results(comparison_result)

    def compare_performance():
        st.text("Comparing the different category performance")

        st.session_state[SESSION_STATE_PERF_COMPARISON_RESULT] = (
            run_performance_comparison(metrics_df_subset=metrics["subgroup_data"])
        )

        # render is triggered which visualizes the result

    st.button(
        "Global significance analysis",
        on_click=compare_performance,
    )
