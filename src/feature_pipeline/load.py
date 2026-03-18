"""
Load & time-split the raw dataset
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_and_split_data(
        raw_path: str = "Y:/code/regression-pipeline/data/raw/HouseTS.csv",
        output_dir: Path | str = DATA_DIR
):
    """Load raw dataset, split into train/val/holdout by date"""
    df = pd.read_csv(raw_path)

    # transforming into parquet
    parquet_path = output_dir / "HouseTs.parquet"
    df.to_parquet(parquet_path)

    # Ensure datetime + sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Cutoffs
    cutoff_date_val = "2020-01-01"
    cutoff_date_holdout = "2022-01-01"

    # Splits
    train_df = df[df["date"] < cutoff_date_val]
    val_df = df[(df["date"] >= cutoff_date_val) & (df["date"] < cutoff_date_holdout)]
    holdout_df = df[df["date"] >= cutoff_date_holdout]

    # saving csv
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)


if __name__ == "__main__":
    load_and_split_data()