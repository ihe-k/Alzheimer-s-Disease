import os
import glob
import pickle
import random
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Optional Streamlit import for GUI mode only
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Explainability
import shap
from lime.lime_tabular import LimeTabularExplainer



# =============================================================================
# 0) CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PREDICT_DATA_DIR", os.path.join(BASE_DIR, "data"))

CSV_DIR = os.path.join(DATA_DIR, "data_csv")
SPLITS = ["train", "valid", "test"]  # your folders are 'train', 'test', 'valid'

# CSV files (PHQ8_Score target lives here)
CSV_TRAIN = os.path.join(CSV_DIR, "train_split_Depression_AVEC2017.csv")
CSV_DEV   = os.path.join(CSV_DIR, "dev_split_Depression_AVEC2017.csv")   # using as "valid"
CSV_TEST  = os.path.join(CSV_DIR, "full_test_split.csv")                 # has PHQ_Score

# Which modalities to use (will load if file exists)
MODALITIES = [
    "audio",
    "fkps",
    "gaze_conf",
    "pose_conf",
    "text",
    # New image modalities (optional, only used if features are prepared)
    "mri",
    "pet",
]



# =============================================================================
# File pattern helpers
# =============================================================================
def candidate_paths(split: str, pid: int, modality: str) -> List[str]:
    paths: List[str] = []
    if split == "train":
        base_dirs = [os.path.join(DATA_DIR, "train")]
        prefixes = ["train"]
    elif split == "valid":
        base_dirs = [os.path.join(DATA_DIR, "valid"), os.path.join(DATA_DIR, "dev")]
        prefixes = ["dev", "valid"]
    elif split == "test":
        base_dirs = [os.path.join(DATA_DIR, "test")]
        prefixes = ["test"]
    else:
        base_dirs = [os.path.join(DATA_DIR, split)]
        prefixes = [split]

    for base in base_dirs:
        for pref in prefixes:
            flat = os.path.join(base, f"{pref}_ft_{modality}_{pid}.npy")
            nested = os.path.join(base, str(pid), f"{pref}_ft_{modality}_{pid}.npy")
            paths.append(flat)
            paths.append(nested)

    # dedupe preserve order
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths

def first_existing(path_list: List[str]) -> Optional[str]:
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

# =============================================================================
# 1b) FEATURE REDUCTION HELPERS
# =============================================================================
MAX_FEATURES_PER_MODALITY = 2048
ONE_D_SEGMENTS = 64

def _downsample_features(vec: np.ndarray, cap: int, base_name: str) -> Tuple[np.ndarray, List[str]]:
    if vec.shape[0] <= cap:
        names = [f"{base_name}_{i}" for i in range(vec.shape[0])]
        return vec.astype(np.float32, copy=False), names
    idx = np.linspace(0, vec.shape[0] - 1, cap, dtype=int)
    vec_ds = vec[idx]
    names = [f"{base_name}_{i}" for i in range(vec_ds.shape[0])]
    return vec_ds.astype(np.float32, copy=False), names

def _summarize_1d(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    arr = np.nan_to_num(arr.astype(np.float32, copy=False))
    stats_vals = [
        float(np.mean(arr)), float(np.std(arr)),
        float(np.min(arr)),  float(np.max(arr)),
    ]
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    try:
        q_vals = np.quantile(arr, quantiles).astype(np.float32)
    except Exception:
        q_vals = np.array([float(np.mean(arr))] * len(quantiles), dtype=np.float32)
    seg_feats: List[float] = []
    if arr.shape[0] > ONE_D_SEGMENTS:
        for seg in np.array_split(arr, ONE_D_SEGMENTS):
            seg_feats.append(float(np.mean(seg)))
    feats = np.concatenate([
        np.array(stats_vals, dtype=np.float32),
        q_vals,
        np.array(seg_feats, dtype=np.float32) if len(seg_feats) else np.array([], dtype=np.float32),
    ])
    names = [f"{modality}_mean", f"{modality}_std", f"{modality}_min", f"{modality}_max"]
    names += [f"{modality}_q{int(q*100)}" for q in quantiles]
    if len(seg_feats):
        names += [f"{modality}_segmean_{i}" for i in range(ONE_D_SEGMENTS)]
    feats, names = _cap_features(feats, names, MAX_FEATURES_PER_MODALITY)
    return feats, names

def _cap_features(vec: np.ndarray, names: List[str], cap: int) -> Tuple[np.ndarray, List[str]]:
    if vec.shape[0] <= cap:
        return vec, names
    idx = np.linspace(0, vec.shape[0] - 1, cap, dtype=int)
    return vec[idx], [names[i] for i in idx.tolist()]

def _summarize_2d_or_more(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    t = arr.shape[0]
    f = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    mat = np.nan_to_num(arr.reshape(t, f).astype(np.float32, copy=False))
    mean_f = np.mean(mat, axis=0)
    std_f = np.std(mat, axis=0)
    feats = np.concatenate([mean_f, std_f])
    names = (
        [f"{modality}_mean_f{i}" for i in range(mean_f.shape[0])] +
        [f"{modality}_std_f{i}" for i in range(std_f.shape[0])]
    )
    if feats.shape[0] > MAX_FEATURES_PER_MODALITY:
        feats, names = _cap_features(feats, names, MAX_FEATURES_PER_MODALITY)
    return feats.astype(np.float32, copy=False), names

def summarize_modality_features(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    if arr.ndim <= 1:
        return _summarize_1d(arr.reshape(-1), modality)
    return _summarize_2d_or_more(arr, modality)

def align_matrix_to_template(X: np.ndarray, current_names: List[str], template_names: List[str]) -> np.ndarray:
    if current_names == template_names:
        return X
    name_to_idx = {n: i for i, n in enumerate(current_names)}
    num_samples = X.shape[0]
    aligned = np.zeros((num_samples, len(template_names)), dtype=X.dtype)
    for j, name in enumerate(template_names):
        i = name_to_idx.get(name)
        if i is not None and i < X.shape[1]:
            aligned[:, j] = X[:, i]
    return aligned


# =============================================================================
# 1) LOAD LABEL TABLES
# =============================================================================
def load_label_table(split: str) -> pd.DataFrame:
    if split == "train":
        df = pd.read_csv(CSV_TRAIN)
        if "PHQ8_Score" not in df.columns:
            raise ValueError("PHQ8_Score not found in train CSV.")
    elif split == "valid":
        df = pd.read_csv(CSV_DEV)
        if "PHQ8_Score" not in df.columns:
            raise ValueError("PHQ8_Score not found in dev CSV.")
    elif split == "test":
        df = pd.read_csv(CSV_TEST)
        if "PHQ8_Score" not in df.columns and "PHQ_Score" in df.columns:
            df = df.rename(columns={"PHQ_Score": "PHQ8_Score", "PHQ_Binary": "PHQ8_Binary"})
    else:
        raise ValueError(f"Unknown split: {split}")
    if "Participant_ID" not in df.columns and "participant_ID" in df.columns:
        df = df.rename(columns={"participant_ID": "Participant_ID"})
    return df

# =============================================================================
# 2) FEATURE LOADING
# =============================================================================
def load_features_for_id(split: str, pid: int) -> Tuple[np.ndarray, List[str]]:
    feats = []
    names = []
    for modality in MODALITIES:
        path = first_existing(candidate_paths(split, pid, modality))
        if path is None:
            continue
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        feat_vec, feat_names_mod = summarize_modality_features(arr, modality)
        feats.append(feat_vec.astype(np.float32, copy=False))
        names.extend(feat_names_mod)

    if len(feats) == 0:
        return np.array([]), []
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False), names

def build_matrix(split: str, use_demographics: bool = True, use_subscores: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    df = load_label_table(split)
    sub_cols = [c for c in df.columns if c.startswith("PHQ8_") and c not in ("PHQ8_Score", "PHQ8_Binary")]
    demo_cols = ["Gender"] if "Gender" in df.columns else []

    X_rows = []
    y_rows = []
    ids = []
    cached_names = None

    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        fvec, fnames = load_features_for_id(split, pid)
        if fvec.size == 0:
            continue

        extra = []
        extra_names: List[str] = []

        if use_demographics and len(demo_cols) > 0:
            for c in demo_cols:
                if c in df.columns and pd.notna(row[c]):
                    extra.append(float(row[c]))
                else:
                    extra.append(0.0)
            extra_names.extend(demo_cols)

        if use_subscores and len(sub_cols) > 0:
            for c in sub_cols:
                val = float(row[c]) if pd.notna(row[c]) and float(row[c]) >= 0 else 0.0
                extra.append(val)
            extra_names.extend(sub_cols)

        full_vec = np.concatenate([fvec, np.array(extra, dtype=float)]) if len(extra) else fvec
        full_names = fnames + extra_names

        if cached_names is None:
            cached_names = full_names
        else:
            if len(full_vec) != len(cached_names):
                max_len = max(len(cached_names), len(full_vec))
                if len(full_vec) < max_len:
                    full_vec = np.pad(full_vec, (0, max_len - len(full_vec)))
                    full_names = full_names + [f"_pad_{i}" for i in range(max_len - len(full_names))]
                if len(cached_names) < max_len:
                    pad_needed = max_len - len(cached_names)
                    cached_names = cached_names + [f"_pad_{i}" for i in range(pad_needed)]
                    X_rows = [np.pad(r, (0, pad_needed)) for r in X_rows]

        X_rows.append(full_vec)
        y_rows.append(float(row["PHQ8_Score"]))
        ids.append(pid)

    if len(X_rows) == 0:
        raise RuntimeError(f"No data found for split={split}. Check file paths and naming.")

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=float)
    feature_names = cached_names if cached_names is not None else [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, ids, feature_names

# =============================================================================
# 3) TRAIN / EVAL
# =============================================================================
def evaluate_model(model: RandomForestRegressor, split_name: str, Xs: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(Xs)
    mae = mean_absolute_error(y, y_pred)
    try:
        rmse = mean_squared_error(y, y_pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = r2_score(y, y_pred)
    print(f"\n==== {split_name.upper()} METRICS ====")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}

def run_full_pipeline() -> Dict[str, Any]:
    # Prepare image features (if mapping CSVs are present)
    # maybe_prepare_image_features()
    print("Loading TRAIN...")
    X_train, y_train, ids_train, feat_names = build_matrix("train", use_demographics=True, use_subscores=True)
    print("Loading VALID...")
    X_valid, y_valid, ids_valid, feat_names_valid = build_matrix("valid", use_demographics=True, use_subscores=True)
    print("Loading TEST...")
    X_test, y_test, ids_test, feat_names_test = build_matrix("test", use_demographics=True, use_subscores=False)

    # Align to train template
    X_valid_al = align_matrix_to_template(X_valid, feat_names_valid, feat_names)
    X_test_al  = align_matrix_to_template(X_test,  feat_names_test,  feat_names)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid_al)
    X_test_s  = scaler.transform(X_test_al)

    # Train model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    m_train = evaluate_model(model, "train", X_train_s, y_train)
    m_valid = evaluate_model(model, "valid", X_valid_s, y_valid)
    m_test  = evaluate_model(model, "test",  X_test_s,  y_test)

    # Save validation plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_valid, m_valid["y_pred"], alpha=0.6)
    lim_min = min(float(np.min(y_valid)), float(np.min(m_valid["y_pred"])))
    lim_max = max(float(np.max(y_valid)), float(np.max(m_valid["y_pred"])))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
    plt.xlabel("True PHQ8 (Valid)")
    plt.ylabel("Predicted PHQ8")
    plt.title("Predicted vs Actual PHQ8 (Valid)")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/validation_plot.png")
    plt.close()

    # Save model artifact (no image data required)
    model_artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feat_names,
        "train_metrics": m_train,
        "valid_metrics": m_valid,
        "test_metrics": m_test,
    }
    model_path = os.path.join("artifacts", "severity_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Saved model: {model_path}")

    # Predict on all
    X_all = np.vstack([X_train, X_valid_al, X_test_al])
    X_all_s = scaler.transform(X_all)
    pred_all = model.predict(X_all_s)

    # Explainability
    explainer_shap = shap.TreeExplainer(model)
    explainer_lime = LimeTabularExplainer(
        X_train_s,
        feature_names=feat_names,
        class_names=["PHQ8_Score"],
        verbose=False,
        mode="regression"
    )

    return {
        "X_train": X_train,
        "X_valid": X_valid_al,
        "X_test": X_test_al,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_valid": ids_valid,
        "ids_test": ids_test,
        "feat_names": feat_names,
        "scaler": scaler,
        "model": model,
        "X_train_s": X_train_s,
        "X_valid_s": X_valid_s,
        "X_test_s": X_test_s,
        "X_all_s": X_all_s,
        "pred_all": pred_all,
        "TOTAL_N": len(ids_train) + len(ids_valid) + len(ids_test),
        "m_train": m_train,
        "m_valid": m_valid,
        "m_test": m_test,
        "explainer_shap": explainer_shap,
        "explainer_lime": explainer_lime,
    }

if __name__ == "__main__":
    results = run_full_pipeline()
    print("Training + evaluation complete!")
    print("Train metrics:", results["m_train"])
    print("Valid metrics:", results["m_valid"])
    print("Test metrics :", results["m_test"])