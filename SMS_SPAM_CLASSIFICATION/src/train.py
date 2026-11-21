"""
Model Training and Cross-Validation Module for SMS Spam Classification.
Trains and compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM
using 5-fold Stratified Cross-Validation, prevents data leakage using Pipelines,
and saves the best model and performance metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from src.data_loader import load_sms_dataset
from src.preprocessing import clean_sms_text, get_custom_stop_words


def build_pipeline(classifier: Any) -> Pipeline:
    """Creates a Scikit-Learn pipeline to prevent data leakage."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_sms_text,
                stop_words=get_custom_stop_words(),
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])


def train_and_compare_models(
    data_path: str,
    models_dir: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Executes full training, 5-fold cross-validation, model comparison,
    model selection based on CV F1-score, test set evaluation, and persistence.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Dataset
    df, stats = load_sms_dataset(data_path, drop_duplicates=True)
    X = df["message"]
    y = df["label_num"]
    
    # 2. Stratified Train-Test Split (80% train, 20% test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    # 3. Algorithms to evaluate
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear Support Vector Machine": LinearSVC(C=1.0, random_state=42)
    }
    
    # 4. Five-Fold Stratified Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }
    
    cv_results = {}
    fitted_pipelines = {}
    test_evaluations = {}
    train_evaluations = {}
    
    for name, clf in candidate_models.items():
        pipe = build_pipeline(clf)
        
        # 5-fold CV
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )
        
        cv_summary = {
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_f1_std": float(np.std(scores["test_f1"])),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(scores["test_precision"])),
            "cv_precision_std": float(np.std(scores["test_precision"])),
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
            "cv_recall_std": float(np.std(scores["test_recall"])),
            "train_f1_mean": float(np.mean(scores["train_f1"])),
            "train_accuracy_mean": float(np.mean(scores["train_accuracy"])),
        }
        cv_results[name] = cv_summary
        
        # Fit on entire 80% train split
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Train split evaluation
        y_train_pred = pipe.predict(X_train)
        train_evaluations[name] = {
            "accuracy": float(accuracy_score(y_train, y_train_pred)),
            "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
            "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
            "f1": float(f1_score(y_train, y_train_pred, zero_division=0))
        }
        
        # Test split evaluation (20% held-out)
        y_test_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Check ROC-AUC capability
        roc_auc = None
        has_proba = hasattr(clf, "predict_proba")
        try:
            if has_proba:
                y_scores = pipe.predict_proba(X_test)[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_scores))
            elif hasattr(clf, "decision_function"):
                y_scores = pipe.decision_function(X_test)
                roc_auc = float(roc_auc_score(y_test, y_scores))
        except Exception:
            roc_auc = None
            
        test_evaluations[name] = {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "has_genuine_proba": has_proba,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "matrix": cm.tolist()
            },
            "classification_report": classification_report(
                y_test,
                y_test_pred,
                target_names=["ham", "spam"],
                output_dict=True
            )
        }
    
    # 5. Model Selection based strictly on Cross-Validation F1-score
    best_model_name = max(
        cv_results.keys(),
        key=lambda k: cv_results[k]["cv_f1_mean"]
    )
    best_pipeline = fitted_pipelines[best_model_name]
    
    # 6. Save Best Pipeline Artifact
    model_save_path = os.path.join(models_dir, "sms_spam_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    
    # 7. Compile Complete Report Metrics
    metrics_report = {
        "dataset_statistics": stats,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_model": best_model_name,
        "selection_criteria": "Highest 5-Fold Stratified Cross-Validation F1-Score",
        "cross_validation_results": cv_results,
        "train_performance": train_evaluations,
        "test_performance": test_evaluations,
        "best_model_test_summary": test_evaluations[best_model_name]
    }
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
        
    return best_model_name, metrics_report


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "..", "dataset", "SMSSpamCollection")
    models_target = os.path.join(current_dir, "..", "models")
    best_name, report = train_and_compare_models(data_file, models_target)
    print("=" * 60)
    print(f"Training Complete. Selected Best Model: {best_name}")
    print(f"CV F1-Score: {report['cross_validation_results'][best_name]['cv_f1_mean']:.4f}")
    print(f"Test F1-Score: {report['test_performance'][best_name]['f1']:.4f}")
    print(f"Test Accuracy: {report['test_performance'][best_name]['accuracy']:.4f}")
    print("=" * 60)
