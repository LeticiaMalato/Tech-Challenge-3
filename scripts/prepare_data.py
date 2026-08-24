"""Data preparation script: cleaning, conflict removal, and splitting.

Runs once. Reads the raw CSVs from the Medical Abstracts TC Corpus dataset
and generates data/processed/{train,val,test}.csv, ready for the rest of the pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = "data"
PROCESSED_DATA_DIR = "data/processed"

URGENCY_MAP = {
    "cardiovascular diseases": "urgent",
    "nervous system diseases": "urgent",
    "neoplasms": "attention",
    "digestive system diseases": "attention",
    "general pathological conditions": "normal",
}


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads the raw train.csv, test.csv, and labels.csv files from disk."""
    train = pd.read_csv(f"{RAW_DATA_DIR}/medical_tc_train.csv")
    test = pd.read_csv(f"{RAW_DATA_DIR}/medical_tc_test.csv")
    labels = pd.read_csv(f"{RAW_DATA_DIR}/medical_tc_labels.csv")
    return train, test, labels


def apply_urgency_mapping(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Adds condition_name and urgency_label columns to a DataFrame."""
    label_names = dict(zip(labels["condition_label"], labels["condition_name"], strict=True))
    df = df.copy()
    df["condition_name"] = df["condition_label"].map(label_names)
    df["urgency_label"] = df["condition_name"].map(URGENCY_MAP)
    return df


def build_pool(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Concatenates train and test into a single pool, tagging origin in 'source'."""
    train = train.copy()
    test = test.copy()
    train["source"] = "train"
    test["source"] = "test"
    return pd.concat([train, test], ignore_index=True)


def remove_conflicting_labels(pool: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicated texts with conflicting labels.

    Duplicated texts whose urgency_label diverges across occurrences are
    dropped entirely; texts with a consistent label keep a single copy.
    """
    label_consistency = pool.groupby("medical_abstract")["urgency_label"].nunique()
    consistent_texts = label_consistency[label_consistency == 1].index
    pool_clean = pool[pool["medical_abstract"].isin(consistent_texts)].copy()
    pool_clean = pool_clean.drop_duplicates(subset="medical_abstract", keep="first")
    return pool_clean


def split_dataset(
    pool_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the clean pool into train/val/test (70/15/15), stratified by urgency_label."""
    train_val, test = train_test_split(
        pool_clean,
        test_size=0.15,
        stratify=pool_clean["urgency_label"],
        random_state=42,
    )
    val_ratio = 0.15 / 0.85
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["urgency_label"],
        random_state=42,
    )
    return train, val, test


def save_processed(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Saves the 3 resulting sets to data/processed/."""
    cols_to_save = ["medical_abstract", "urgency_label", "source"]
    train[cols_to_save].to_csv(f"{PROCESSED_DATA_DIR}/train.csv", index=False)
    val[cols_to_save].to_csv(f"{PROCESSED_DATA_DIR}/val.csv", index=False)
    test[cols_to_save].to_csv(f"{PROCESSED_DATA_DIR}/test.csv", index=False)


def main() -> None:
    """Orchestrates the data preparation pipeline, from raw to processed."""
    train_raw, test_raw, labels = load_raw_data()
    train_mapped = apply_urgency_mapping(train_raw, labels)
    test_mapped = apply_urgency_mapping(test_raw, labels)
    pool = build_pool(train_mapped, test_mapped)
    pool_clean = remove_conflicting_labels(pool)
    train, val, test = split_dataset(pool_clean)
    save_processed(train, val, test)
    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")


if __name__ == "__main__":
    main()
