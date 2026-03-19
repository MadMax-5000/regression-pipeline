"""
valuate a saved XGBoost model on the val split.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_VAL = Path("data/processed/fe_val.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def valuate_model(
    model_path: Path | str = DEFAULT_MODEL,
    val_path: Path | str = DEFAULT_VAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    val_df = pd.read_csv(val_path)
    val_df = _maybe_sample(val_df, sample_frac, random_state)

    target = "price"
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]

    model = load(model_path)
    y_pred = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("📊 Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    valuate_model()

