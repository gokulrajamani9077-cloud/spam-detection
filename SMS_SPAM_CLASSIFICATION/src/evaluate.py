"""
Evaluation and Visualization Module for SMS Spam Classification.
Generates metrics, classification reports, confusion matrix visualisations,
and lexical distribution plots for EDA and viva presentation.
"""

import os
from collections import Counter
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.preprocessing import clean_sms_text, get_custom_stop_words


def plot_class_distribution(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Plot 1: Ham vs Spam Class Distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["label"].value_counts()
    colors = ["#2E7D32", "#C62828"]
    
    bars = ax.bar(counts.index.str.upper(), counts.values, color=colors, width=0.5, edgecolor="black")
    ax.set_title("SMS Class Distribution (Ham vs Spam)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("SMS Category", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Messages", fontsize=12, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Annotate bar counts and percentages
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_message_length_distribution(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Plot 2: Message Length Distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="char_length",
        hue="label",
        bins=50,
        kde=True,
        palette={"ham": "#2E7D32", "spam": "#C62828"},
        ax=ax,
        alpha=0.6
    )
    ax.set_title("SMS Character Length Distribution by Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Message Character Length", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)
    ax.set_xlim(0, 300)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_word_count_distribution(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Plot 3: Word Count Distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="word_count",
        hue="label",
        bins=40,
        kde=True,
        palette={"ham": "#2E7D32", "spam": "#C62828"},
        ax=ax,
        alpha=0.6
    )
    ax.set_title("SMS Word Count Distribution by Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Total Words per Message", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)
    ax.set_xlim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def get_top_frequent_words(texts: pd.Series, top_n: int = 15) -> List[tuple]:
    """Helper to tokenize, clean, remove custom stop words, and get top tokens."""
    stop_words_set = set(get_custom_stop_words())
    words = []
    for msg in texts:
        cleaned = clean_sms_text(msg)
        tokens = cleaned.split()
        for tok in tokens:
            tok_clean = "".join([c for c in tok if c.isalnum()])
            if len(tok_clean) > 2 and tok_clean not in stop_words_set:
                words.append(tok_clean)
    return Counter(words).most_common(top_n)


def plot_frequent_words(df: pd.DataFrame, target_label: str, output_path: Optional[str] = None) -> plt.Figure:
    """Plot 4 & 5: Most Frequent Words in Spam or Ham Messages."""
    subset_texts = df[df["label"] == target_label]["message"]
    top_words = get_top_frequent_words(subset_texts, top_n=15)
    
    words, freqs = zip(*top_words) if top_words else ([], [])
    color = "#C62828" if target_label == "spam" else "#2E7D32"
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(words))
    ax.barh(y_pos, freqs, color=color, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis()  # Top word at the top
    ax.set_xlabel("Word Occurrence Frequency", fontsize=12, labelpad=10)
    ax.set_title(f"Top 15 Most Frequent Words in {target_label.upper()} Messages", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    for i, v in enumerate(freqs):
        ax.text(v + 1, i, str(v), va="center", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_confusion_matrix_heatmap(
    cm_matrix: list,
    model_name: str,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot 6: Confusion Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm_matrix)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted HAM", "Predicted SPAM"],
        yticklabels=["Actual HAM", "Actual SPAM"],
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def explain_confusion_matrix(cm_dict: Dict[str, int]) -> Dict[str, str]:
    """Returns clear academic explanations of TP, TN, FP, and FN."""
    tn = cm_dict.get("true_negative", 0)
    fp = cm_dict.get("false_positive", 0)
    fn = cm_dict.get("false_negative", 0)
    tp = cm_dict.get("true_positive", 0)
    
    return {
        "True Negative (TN)": f"{tn} messages: Legitimate (HAM) messages correctly classified as HAM. Safe messages received undisturbed.",
        "False Positive (FP)": f"{fp} messages: Legitimate (HAM) messages incorrectly flagged as SPAM. Known as Type I error; causes user inconvenience.",
        "False Negative (FN)": f"{fn} messages: Malicious (SPAM) messages erroneously marked as HAM. Known as Type II error; poses security risk.",
        "True Positive (TP)": f"{tp} messages: Actual SPAM messages accurately intercepted and identified by the classifier."
    }
