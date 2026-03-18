"""
Preprocessing script for Housing regression MLE

- reads train/val/holdout CSVs from data/raw
- cleans and normalize city names
- maps cities to metros and merges lat/lng
- drop duplicates and outliers
- saves cleaned splits to data/processed
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# city mapping fixes 
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}

def normalize_city(s: str) -> str:
    """lowercase, strip, unify dashes"""
    if pd.isna(s):
        return s 
    s = str(s).strip().lower()
    s = re.sub(r"[–—-]", "-", s)          # unify dashes
    s = re.sub(r"\s+", " ", s)            # collapse spaces
    return s

def clean_and_merge(df: pd.DataFrame, metros_path: str | None = "data/raw/usmetros.csv"):
    """Normalize city names and merge lat/lng from metros dataset"""

    if "city_full" not in df.columns:
        print("Skipping city merge: no 'city_full' column present")
        return df
    
    df["city_full"] = df["city_full"].apply(normalize_city)

    # applying mapping
    norm_mapping = {normalize_city(k) : normalize_city(v) for k,v in CITY_MAPPING.items()}
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # if lat and lng alredy in dataframe
    if {"lat", "lng"}.issubset(df.columns):
        print("lat and lng already in df")
        return df

    # if no metros file provided/exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        print("Skipping merge: metros file not provided or not found.")
        return df
    
    # merge lat/lng
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("skipping merge: metros file missing required columns")
        return df
    
    metros["metro_full"] = metros["metro_full"].apply(normalize_city)

    df = df.merge(metros[["metro_full", "lat", "lng"]], how="left", left_on="city_full", right_on="metro_full")
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    # handling missing columns
    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print(f"Still missing lat/lng for {missing}")
    else:
        print("All cities matched with metros dataset")
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years"""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f"Dropped {after - before} duplicated rows (excluding date and year columns)")
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers in median_list_price where >19M """
    if "median_list_price" not in df.columns:
        return df
    
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print("Removed all houses with a price > 19M")
    return df

def preprocess_split(
        split: str,
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path |  str = PROCESSED_DIR,
        metros_path: Path | None = "data/raw/usmetros.csv"
) -> pd.DataFrame:
    
    """Run preprocessing for a split and save the processed datasets"""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir) 
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Processed {split} saved to {out_path} ({df.shape})")

    return df

def run_preprocesses(
    splits: tuple[str, ...] = ("train", "val", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: Path | None = "data/raw/usmetros.csv"
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    run_preprocesses()

