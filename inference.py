import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple

# -------------------------------
# Config
# -------------------------------
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "severity_model.pkl")
DATA_DIR = "data"   # adjust if needed
MODALITIES = ["audio", "fkps", "gaze_conf", "pose_conf", "text", "mri", "pet"]

# -------------------------------
# Load trained model
# -------------------------------
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
scaler = artifact["scaler"]
feat_names_template = artifact["feature_names"]

# -------------------------------
# Utilities
# -------------------------------
def candidate_paths(split: str, pid: int, modality: str) -> List[str]:
    """Return possible feature file paths for a participant+modality."""
    flat = os.path.join(DATA_DIR, split, f"{split}_ft_{modality}_{pid}.npy")
    nested = os.path.join(DATA_DIR, split, str(pid), f"{split}_ft_{modality}_{pid}.npy")
    return [flat, nested]

def first_existing(path_list: List[str]) -> str:
    """Return first existing path, else None."""
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

def summarize_modality_features(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    """Minimal summarization: flatten if >1D, else keep."""
    arr = np.nan_to_num(arr.astype(np.float32))
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr, [f"{modality}_{i}" for i in range(arr.shape[0])]

def align_to_template(vec: np.ndarray, names: List[str], template_names: List[str]) -> np.ndarray:
    """Align feature vector to training template."""
    aligned = np.zeros(len(template_names), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(names)}
    for j, name in enumerate(template_names):
        i = name_to_idx.get(name)
        if i is not None and i < len(vec):
            aligned[j] = vec[i]
    return aligned

def load_features_for_pid(split: str, pid: int) -> np.ndarray:
    """Load & align features for one participant."""
    feats, names = [], []
    for modality in MODALITIES:
        path = first_existing(candidate_paths(split, pid, modality))
        if path is None:
            continue
        arr = np.load(path, allow_pickle=False)
        vec, fnames = summarize_modality_features(arr, modality)
        feats.append(vec)
        names.extend(fnames)
    if len(feats) == 0:
        raise RuntimeError(f"No features found for PID={pid}")
    full_vec = np.concatenate(feats, axis=0)
    return align_to_template(full_vec, names, feat_names_template)

# -------------------------------
# Inference function
# -------------------------------
def predict_for_ids(split: str, ids: List[int]) -> pd.DataFrame:
    """Run inference on a list of participant IDs."""
    X_rows = []
    for pid in ids:
        try:
            vec = load_features_for_pid(split, pid)
            X_rows.append(vec)
        except RuntimeError as e:
            print(f"Skipping {pid}: {e}")
            continue
    if not X_rows:
        raise RuntimeError("No valid participants found for inference.")
    X = np.vstack(X_rows)
    X_s = scaler.transform(X)
    preds = model.predict(X_s)
    return pd.DataFrame({"Participant_ID": ids, "Predicted_PHQ8": preds})

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example: predict for participants 100 and 200 in the "test" split
    participant_ids = [300, 301]
    results = predict_for_ids("test", participant_ids)
    print(results)
    results.to_csv("artifacts/inference_results.csv", index=False)
    print("Saved predictions → artifacts/inference_results.csv")
