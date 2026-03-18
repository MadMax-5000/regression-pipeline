"""
Feature engineering : date parts, zipcode frequency encoding, target encoding, drop leakage
- Reads cleaned train/val CSVs
- Applies feature engineering
- Saves feature engineered CSVs
- Saves fitted encoders for inference
"""

from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    # place after date for readability (optional)
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df

def frequency_encode(train:pd.DataFrame, val:pd.DataFrame, col:str):
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    val[f"{col}_freq"] = val[col].map(freq_map).fillna(0)
    return train, val, freq_map

# uses target encoding 
def target_encode(train: pd.DataFrame, val: pd.DataFrame, col:str, target:str):
    """Use target encoding on 'col'"""
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[col], train[target])
    val[encoded_col] = te.transform(val[col])
    return train, val, te

def drop_unused_columns(train: pd.DataFrame, val:pd.DataFrame):
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train.drop(columns=[c for c in drop_cols if c in train.columns], inplace=True, errors="ignore")
    val.drop(columns=[c for c in drop_cols if c in val.columns], inplace=True, errors="ignore")
    return train, val

# PIPELINE
# Handles full pipeline: 
# reads cleaned CSVs → applies feature engineering → saves engineered data + encoders

def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_val_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """Run fe and writes outputs + encoders to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_val_path is None:
        in_val_path = PROCESSED_DIR / "cleaning_val.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    val_df = pd.read_csv(in_val_path)
    holdout_df = pd.read_csv(in_holdout_path)
    
    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Val date range:", val_df["date"].min(), "to", val_df["date"].max())
    print("Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max())

    # Date features
    train_df = add_date_features(train_df)
    val_df = add_date_features(val_df)
    holdout_df = add_date_features(holdout_df)

    # Frequency encode zipcode
    freq_map = None
    if "zip_code" in train_df.columns:
        train_df, val_df, holdout_df = target_encode(train_df, val_df, "zipcode")
        holdout_df["zipcode_freqs"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl") # save mapping
    
    # drop leakage
    train_df, val_df = drop_unused_columns(train_df, val_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # save engineered data
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_val_path = output_dir / "feature_engineered_val.csv"
    out_holdout_path = output_dir / "feature_engineered_holdout.csv"
    train_df.to_csv(out_train_path, index=False)
    val_df.to_csv(out_val_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Val  shape:", val_df.shape)
    print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, val_df, holdout_df, freq_map, target_encode


if __name__ == "__main__":
    run_feature_engineering()