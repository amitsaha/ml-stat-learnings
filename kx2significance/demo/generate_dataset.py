import numpy as np
import pandas as pd


def generate_dataset(
    n_rows=1000,
    group_defs=None,
    seed=42,
):
    """
    group_defs: dict of {column_name: [possible_values]}
      e.g.
      {
        "group:gender": ["F", "M"],
        "group:ethnicity": ["WHITE", "BLACK", "ASIAN", "OTHER"],
        "group:age": ["age_<40", "age_40_65", "age_65+"],
      }
    """
    rng = np.random.default_rng(seed)

    if group_defs is None:
        group_defs = {
            "group:gender": ["F", "M"],
            "group:ethnicity": ["WHITE", "BLACK", "ASIAN", "OTHER"],
        }

    df = pd.DataFrame()

    # ---- labels ----
    df["label"] = rng.integers(0, 2, size=n_rows)

    # ---- probability scores ----
    # mildly correlated with label (optional but realistic)
    base_score = rng.beta(2, 5, size=n_rows)
    df["score"] = np.clip(
        base_score + 0.4 * df["label"],
        0.0,
        1.0,
    )

    # ---- group columns ----
    for col, values in group_defs.items():
        df[col] = rng.choice(values, size=n_rows)

    return df


def apply_white_preference_bias(
    df: pd.DataFrame,
    ethnicity_col: str = "group:ethnicity",
    favored_group: str = "WHITE",
    score_col: str = "score",
    label_col: str = "label",
    uplift: float = 0.25,
    noise: float = 0.05,
    seed: int = 42,
):
    """
    Increase predicted scores for the favored ethnicity group,
    creating higher positive prediction rates post-threshold.

    Parameters
    ----------
    uplift : float
        Additive score boost for favored group when label == 1
    noise : float
        Small Gaussian noise added to all scores
    """
    rng = np.random.default_rng(seed)

    df = df.copy()

    # baseline noise (keeps realism)
    df[score_col] += rng.normal(0.0, noise, size=len(df))

    favored_mask = df[ethnicity_col] == favored_group
    positive_mask = df[label_col] == 1

    # preferential uplift only for true positives in WHITE group
    df.loc[favored_mask & positive_mask, score_col] += uplift

    # clip to valid probability range
    df[score_col] = df[score_col].clip(0.0, 1.0)

    return df


df = generate_dataset(
    n_rows=5000,
    group_defs={
        "group:gender": ["F", "M"],
        "group:ethnicity": ["WHITE", "BLACK", "ASIAN", "HISPANIC"],
        "group:age": ["18-39", "40-64", "65+"],
    },
)

df.to_csv("dataset.csv.tar.gz")


df_biased = apply_white_preference_bias(
    df,
    uplift=0.3,
)

df_biased.to_csv("dataset_white_biased.csv.tar.gz")
