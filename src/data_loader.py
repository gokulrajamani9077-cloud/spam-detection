"""
Data Loader Module for SMS Spam Classification System
Handles loading, validation, and exploration statistics of the UCI SMSSpamCollection dataset.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def load_sms_dataset(
    filepath: str,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and parse the original UCI SMS Spam Collection file.
    
    The raw file is tab-separated with no header row:
    Column 0: 'ham' or 'spam'
    Column 1: Raw SMS message text
    
    Args:
        filepath: Path to SMSSpamCollection file
        drop_duplicates: Whether to remove duplicate SMS entries
        
    Returns:
        df: Processed pandas DataFrame with columns ['label', 'message', 'label_num', 'char_length', 'word_count']
        stats: Summary statistics dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SMS dataset not found at: {filepath}")

    # Read tab-separated file with encoding fallback
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["label", "message"],
            quoting=3,
            encoding="utf-8"
        )
    except (UnicodeDecodeError, Exception):
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["label", "message"],
            quoting=3,
            encoding="latin-1"
        )

    # Initial shape
    total_raw = len(df)
    
    # Clean whitespace and drop nulls
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])].dropna(subset=["message"])
    
    # Missing value count
    missing_count = total_raw - len(df)
    
    # Detect duplicates
    duplicate_count = int(df.duplicated(subset=["message"]).sum())
    
    if drop_duplicates:
        df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
        
    # Derived analytical columns
    df["label_num"] = (df["label"] == "spam").astype(int)
    df["char_length"] = df["message"].str.len()
    df["word_count"] = df["message"].apply(lambda txt: len(txt.split()))
    
    clean_total = len(df)
    ham_count = int((df["label"] == "ham").sum())
    spam_count = int((df["label"] == "spam").sum())
    spam_percentage = round((spam_count / clean_total) * 100, 2) if clean_total > 0 else 0.0

    stats = {
        "total_raw": total_raw,
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "clean_total": clean_total,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "spam_percentage": spam_percentage,
        "duplicates_dropped": drop_duplicates,
        "avg_ham_length": round(float(df[df["label"] == "ham"]["char_length"].mean()), 2),
        "avg_spam_length": round(float(df[df["label"] == "spam"]["char_length"].mean()), 2),
        "avg_ham_words": round(float(df[df["label"] == "ham"]["word_count"].mean()), 2),
        "avg_spam_words": round(float(df[df["label"] == "spam"]["word_count"].mean()), 2),
    }

    return df, stats


if __name__ == "__main__":
    # Self-test script execution
    sample_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "SMSSpamCollection")
    if os.path.exists(sample_path):
        data, info = load_sms_dataset(sample_path)
        print("Data loaded successfully!")
        print(f"Total entries: {info['clean_total']} (Ham: {info['ham_count']}, Spam: {info['spam_count']})")
        print(f"Spam ratio: {info['spam_percentage']}%")
