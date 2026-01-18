from collections import defaultdict
import os
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.stats.contingency_tables import Table
from scipy.stats import chi2_contingency

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


LABEL = "label"
POSITIVE_PROBABILITY_SCORE = "score"
CATEGORY_PREFIX = "group:"


def calculate_metrics(df, prob_threshold):
    y_true = df[LABEL]
    y_proba = df[POSITIVE_PROBABILITY_SCORE]
    y_pred = y_proba > prob_threshold

    if y_true.empty or y_proba.empty:
        return np.nan

    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm_data": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# metrics is an object with:
# key - subgroup
# value -  tp, fn, tn, fp
def compare_categories(metrics, seed=42):
    print("Running global performance analysis with:")
    analysis_result = {}
    # collect all tp and fn
    all_tp = []
    all_fn = []
    for item in metrics.values():
        all_tp.append(item["cm_data"]["tp"])
        all_fn.append(item["cm_data"]["fn"])

    tp_fn = [(k, v) for k, v in zip(all_tp, all_fn)]

    table = np.array(tp_fn)
    print(table)

    try:
        obs_stat, p_value = ffh_monte_carlo(table=table, seed=seed)
        kx2_significant = p_value < 0.05
    except ValueError:
        # if there are any zero rows
        print(table)
        kx2_significant = np.nan

    analysis_result = {
        "global_significant": kx2_significant,
    }
    return analysis_result


def ffh_monte_carlo(table, reps=100_000, seed=42):
    """
    Monte Carlo Fisher–Freeman–Halton test for Kx2 table
    """
    rng = np.random.default_rng(seed)

    # Observed chi-square statistic
    obs_stat, _, _, _ = chi2_contingency(table, correction=False)

    # lock in the table properties
    # individual row and column margins

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    exceed = 0

    for _ in range(reps):
        # Sample under null: hypergeometric allocation
        simulated = np.zeros_like(table)

        remaining_col_sum = col_sums.copy()

        for i, r in enumerate(row_sums[:-1]):
            x = rng.hypergeometric(remaining_col_sum[0], remaining_col_sum[1], r)
            simulated[i, 0] = x
            simulated[i, 1] = r - x
            remaining_col_sum -= simulated[i]

        simulated[-1] = remaining_col_sum

        stat, _, _, _ = chi2_contingency(simulated, correction=False)
        if stat >= obs_stat:
            exceed += 1

    p_value = (exceed + 1) / (reps + 1)
    return obs_stat, p_value


def get_result_prefixes(data_groups):
    return data_groups


def plot_recall_trends(metrics):
    # print(metrics_df)
    # print(metrics_df.shape)

    # data bins/groups (>=1)

    overall_data = metrics["overall_data"]
    subgroup_data = metrics["subgroup_data"]

    # subgroup categories (group within each category)
    subgroup_categories = subgroup_data.keys()
    # print(subgroup_categories)

    # overall recalls
    overall_recall = overall_data["recall"]

    fig = plt.figure(figsize=(12, 6))

    subgroup_colors = {
        e: c
        for e, c in zip(
            subgroup_categories, plt.cm.tab10.colors[: len(subgroup_categories)]
        )
    }

    # Collect subgroup recalls
    subgroup_recalls: Dict[str, List[float]] = {e: [] for e in subgroup_categories}

    for e in subgroup_categories:
        recall = subgroup_data[e].get("recall")
        if not recall:
            print(f"No recall data found for {e}")
            subgroup_recalls[e] = np.nan
        else:
            subgroup_recalls[e] = recall

    overall_color = "black"

    # Plot overall (black line)
    plt.plot(
        0,
        overall_recall,
        marker="o",
        linewidth=2,
        label="overall",
        color=overall_color,
    )

    x_pos = 1

    for e, recall in subgroup_recalls.items():
        print(e, recall)
        x_pos += 1
        # print(recalls)
        plt.plot(
            x_pos,
            recall,
            # linestyle=style,
            linewidth=1.5,
            marker="x",
            # color=subgroup_colors[e],
            label=e,
        )

    plt.xlabel("Subgroup")
    plt.ylabel("Recall")
    # plt.xticks(["overall"] + list(subgroup_categories), rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    return fig


def plot_f1_trends(metrics):
    # print(metrics_df)
    # print(metrics_df.shape)

    # data bins/groups (>=1)

    overall_data = metrics["overall_data"]
    subgroup_data = metrics["subgroup_data"]

    # subgroup categories (group within each category)
    subgroup_categories = subgroup_data.keys()
    # print(subgroup_categories)

    # overall recalls
    overall_recall = overall_data["f1"]

    fig = plt.figure(figsize=(12, 6))

    subgroup_colors = {
        e: c
        for e, c in zip(
            subgroup_categories, plt.cm.tab10.colors[: len(subgroup_categories)]
        )
    }

    # Collect subgroup recalls
    subgroup_recalls: Dict[str, List[float]] = {e: [] for e in subgroup_categories}

    for e in subgroup_categories:
        recall = subgroup_data[e].get("f1")
        if not recall:
            print(f"No recall data found for {e}")
            subgroup_recalls[e] = np.nan
        else:
            subgroup_recalls[e] = recall

    overall_color = "black"

    # Plot overall (black line)
    plt.plot(
        0,
        overall_recall,
        marker="o",
        linewidth=2,
        label="overall",
        color=overall_color,
    )

    x_pos = 1

    for e, recall in subgroup_recalls.items():
        print(e, recall)
        x_pos += 1
        # print(recalls)
        plt.plot(
            x_pos,
            recall,
            # linestyle=style,
            linewidth=1.5,
            marker="x",
            # color=subgroup_colors[e],
            label=e,
        )

    plt.xlabel("Subgroup")
    plt.ylabel("Recall")
    # plt.xticks(["overall"] + list(subgroup_categories), rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    return fig


# incoming data is:
# key-value pairs (dataframe/dict)
# key -> data group (>=1)
# value:
# dict of key-value pairs
# key: (pair1, pair2)
# value: {diff_point_estimate: xx, diff_ci: (p0, p1), significant: true/false}

markers = ["v", "o", "p", "^", "H", "X", "8"]


def plot_pairwise_recall(data):
    # print(data)
    data_groups = data.keys()

    # if we have more than one data group, we use the bin position (x-axis) based on the data group
    # it will be tricky, and make the graph pretty much unreadable but we can consider it a UX problem where we
    # can have may be one pair and show it across all data groups
    # if we have one data group, we will simply use the pair position as the bin pos

    bin_pos = []
    point_estimates = []
    point_estimate_errors = []
    diff_is_significant = []
    for idx, d in enumerate(data_groups):
        bin_pos.append(idx)

        group_data = data[d]
        labels = []
        # print(group_data)
        for k, v in group_data.items():
            point_estimate = v["recall_diff"]
            # print(k, v)
            labels.append(k)
            point_estimates.append(point_estimate)
            if not np.isnan(point_estimate):
                ci_lower, ci_upper = v["recall_diff_estimate"]
                lower_error = point_estimate - ci_lower
                upper_error = ci_upper - point_estimate
                point_estimate_errors.append(
                    (
                        lower_error,
                        upper_error,
                    )
                )
                diff_is_significant.append(v["significant"])
            else:
                # for nan
                point_estimate_errors.append((0, 0))
                diff_is_significant.append(False)

    # override of bin pos
    if len(data_groups) == 1:
        bin_pos = range(len(point_estimates))

    point_estimate_errors = np.array(point_estimate_errors)
    point_estimate_errors = point_estimate_errors.T

    fig = plt.figure(figsize=(12, 6))

    # desired error = 0
    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1.2,
        color="grey",
        label="diff = 0",
    )

    plt.errorbar(
        bin_pos,  # array of data positions
        point_estimates,  # array of diff point estimates
        yerr=point_estimate_errors,  # array of error bars (2, N) (first row: lower errors, second row: upper errors)
        marker=markers[0],
        # mfc=subgroup_colors[i],  # marker facecolor
        # mec=subgroup_colors[i],  # marker edge color
        capsize=0.2,
        # ecolor=subgroup_colors[i],
    )

    # add pair data along with if the difference is significant
    ax = plt.gca()

    for idx in bin_pos:
        point_estimate = point_estimates[idx]
        is_significant = diff_is_significant[idx]
        if np.isnan(point_estimate):
            point_estimate = 0

        label = labels[idx]
        if is_significant:
            label = f"{label}*"

        ax.text(
            bin_pos[idx],
            point_estimate,
            label,
            ha="center",
            va="top",
            horizontalalignment="center",
            color="red",
            fontsize=7,
            clip_on=False,
        )

    return fig


# for demos only
def ffh_monte_carlo_demo(table, reps=100_000, seed=42):
    """
    Monte Carlo Fisher–Freeman–Halton test for Kx2 table
    """
    rng = np.random.default_rng(seed)

    # Observed chi-square statistic
    obs_stat, _, _, _ = chi2_contingency(table, correction=False)

    # lock in the table properties
    # individual row and column margins

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    exceed = 0

    simulated_tables = []

    for _ in range(reps):
        # Sample under null: hypergeometric allocation
        simulated = np.zeros_like(table)

        remaining_col = col_sums.copy()

        for i, r in enumerate(row_sums[:-1]):
            x = rng.hypergeometric(remaining_col[0], remaining_col[1], r)
            simulated[i, 0] = x
            simulated[i, 1] = r - x
            remaining_col -= simulated[i]

        simulated[-1] = remaining_col

        simulated_tables.append(simulated)

        stat, _, _, _ = chi2_contingency(simulated, correction=False)
        if stat >= obs_stat:
            exceed += 1

    p_value = (exceed + 1) / (reps + 1)
    return obs_stat, p_value, simulated_tables


# for demos only
def fisher_monte_carlo_convergence(
    table,
    reps=100_000,
    checkpoints=(100, 500, 1_000, 5_000, 10_000, 100_000),
    seed=42,
):
    rng = np.random.default_rng(seed)

    obs_stat, _, _, _ = chi2_contingency(table, correction=False)

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    exceed = 0
    trace = []

    for rep in range(reps + 1):
        # Sample under null: hypergeometric allocation
        simulated = np.zeros_like(table)

        remaining_col = col_sums.copy()

        for i, r in enumerate(row_sums[:-1]):
            x = rng.hypergeometric(remaining_col[0], remaining_col[1], r)
            simulated[i, 0] = x
            simulated[i, 1] = r - x
            remaining_col -= simulated[i]

        simulated[-1] = remaining_col

        stat, _, _, _ = chi2_contingency(simulated, correction=False)
        if stat >= obs_stat:
            exceed += 1

        if rep in checkpoints:
            p_hat = (exceed + 1) / rep
            trace.append({"reps": rep, "p_value": p_hat, "exceed": exceed})

    print(trace)
    return trace
