import pandas as pd
import numpy as np
from typing import Tuple

def prepare_disaster_dataset(
    data_path: str,
    sequence_path: str,
    disaster_code: str,
    drop_columns: list[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:

    df = pd.read_csv(data_path)

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    drop_cols += drop_columns or []
    df = df.drop(columns=drop_cols, errors="ignore")

    code_mask = df["natural_signs_used"].astype(str).str.startswith(disaster_code)
    df = df[code_mask].reset_index(drop=True)

    all_has_cols = [c for c in df.columns if c.startswith("has_")]
    relevant_cols = [c for c in all_has_cols if c.startswith(f"has_{disaster_code}")]
    drop_irrelevant = [c for c in all_has_cols if c not in relevant_cols]
    df = df.drop(columns=drop_irrelevant, errors="ignore")

    df = df.drop(columns=["natural_signs_used", "coast_name"], errors="ignore")

    df[f"{disaster_code.lower()}_pattern_ratio"] = (
        df["num_matched_rules"] / len(relevant_cols)
        if len(relevant_cols) > 0
        else np.nan
    )

    df_seq = pd.read_csv(sequence_path)
    df_seq = df_seq[df_seq["pattern"].str.startswith(f"{disaster_code}-")].copy()

    df_seq["col_name"] = "has_" + df_seq["pattern"].str.replace(",", "_")
    df_seq["mean_prob_norm"] = df_seq["mean_prob"] / 100.0

    weights = dict(zip(df_seq["col_name"], df_seq["mean_prob_norm"]))
    available_cols = [c for c in weights.keys() if c in df.columns]

    if available_cols:
        df[f"{disaster_code.lower()}_weighted_sum"] = sum(
            df[c] * weights[c] for c in available_cols
        )
        max_val = df[f"{disaster_code.lower()}_weighted_sum"].max()
        df[f"{disaster_code.lower()}_weighted_sum_norm"] = (
            df[f"{disaster_code.lower()}_weighted_sum"] / max_val if max_val > 0 else 0
        )
        df = df.drop(columns=[f"{disaster_code.lower()}_weighted_sum"], errors="ignore")
    else:
        df[f"{disaster_code.lower()}_weighted_sum_norm"] = np.nan

    df = df.drop(columns=["risk_score"], errors="ignore")

    X = df.drop(columns=["risk_label"], errors="ignore")
    y = df["risk_label"]

    print(f"Dataset for {disaster_code} prepared: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

import pandas as pd
import numpy as np
from typing import Tuple

import pandas as pd
import numpy as np
from typing import Tuple

def prepare_all_disasters_dataset(
    data_path: str,
    sequence_path: str,
    drop_columns: list[str] = None,
    agg_method: str = "mean"  # bisa 'mean', 'max', atau 'sum'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess dataset for all disaster types (Wn, Ts, Cr, etc.)
    Aggregates weighted_sum_norm and pattern_ratio into unified columns.

    Parameters
    ----------
    data_path : str
        Path to the main dataset (CSV).
    sequence_path : str
        Path to the sequence mining output (CSV).
    drop_columns : list[str], optional
        Additional columns to drop from the main dataset.
    agg_method : str, optional
        How to combine multi-disaster features: 'mean', 'max', or 'sum'.

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe with combined engineered features.
    y : pd.Series
        Target labels.
    """

    # ============================
    # 1️⃣ Load & clean base dataset
    # ============================
    df = pd.read_csv(data_path)
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    drop_cols += drop_columns or []
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.drop(columns=["coast_name"], errors="ignore")

    # ============================
    # 2️⃣ Identify all disaster prefixes
    # ============================
    all_has_cols = [c for c in df.columns if c.startswith("has_")]
    disaster_prefixes = sorted({col.split("_")[1].split("-")[0] for col in all_has_cols})
    print(f"Detected disaster types: {disaster_prefixes}")

    # ============================
    # 3️⃣ Load sequence mining results
    # ============================
    df_seq = pd.read_csv(sequence_path)
    df_seq["col_name"] = "has_" + df_seq["pattern"].str.replace(",", "_")
    df_seq["mean_prob_norm"] = df_seq["mean_prob"] / 100.0

    # Store intermediate feature names
    ratio_cols, weight_cols = [], []

    # ============================
    # 4️⃣ Process each disaster group
    # ============================
    for disaster_code in disaster_prefixes:
        relevant_cols = [c for c in all_has_cols if c.startswith(f"has_{disaster_code}")]
        if not relevant_cols:
            continue

        # --- pattern ratio ---
        ratio_col = f"{disaster_code.lower()}_pattern_ratio"
        df[ratio_col] = (
            df["num_matched_rules"] / len(relevant_cols)
            if len(relevant_cols) > 0
            else np.nan
        )
        ratio_cols.append(ratio_col)

        # --- weighted sum ---
        df_seq_sub = df_seq[df_seq["pattern"].str.startswith(f"{disaster_code}-")].copy()
        weights = dict(zip(df_seq_sub["col_name"], df_seq_sub["mean_prob_norm"]))
        available_cols = [c for c in weights.keys() if c in df.columns]

        weight_col = f"{disaster_code.lower()}_weighted_sum_norm"

        if available_cols:
            weighted_sum = sum(df[c] * weights[c] for c in available_cols)
            max_val = weighted_sum.max()
            df[weight_col] = weighted_sum / max_val if max_val > 0 else 0
        else:
            df[weight_col] = np.nan

        weight_cols.append(weight_col)

    # ============================
    # 5️⃣ Aggregate pattern_ratio & weighted_sum_norm
    # ============================
    if ratio_cols:
        if agg_method == "sum":
            df["pattern_ratio"] = df[ratio_cols].sum(axis=1)
        elif agg_method == "max":
            df["pattern_ratio"] = df[ratio_cols].max(axis=1)
        else:
            df["pattern_ratio"] = df[ratio_cols].mean(axis=1)
        df = df.drop(columns=ratio_cols, errors="ignore")

    if weight_cols:
        if agg_method == "sum":
            df["weighted_sum_norm"] = df[weight_cols].sum(axis=1)
        elif agg_method == "max":
            df["weighted_sum_norm"] = df[weight_cols].max(axis=1)
        else:
            df["weighted_sum_norm"] = df[weight_cols].mean(axis=1)
        df = df.drop(columns=weight_cols, errors="ignore")

    # ============================
    # 6️⃣ Final cleanup
    # ============================
    df = df.drop(columns=["risk_score", "natural_signs_used"], errors="ignore")
    X = df.drop(columns=["risk_label"], errors="ignore")
    y = df["risk_label"]

    print(f"\n✅ Combined dataset ready ({agg_method}-aggregated): {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

