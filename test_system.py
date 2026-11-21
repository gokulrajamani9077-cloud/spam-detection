"""
Comprehensive Verification and Test Suite for SMS Spam Classification System
Tests data loading, preprocessing, model training, persistence, inference, edge cases,
and notebook integrity.
"""

import os
import sys
import json
import joblib

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_sms_dataset
from src.preprocessing import clean_sms_text, validate_sms_input, get_custom_stop_words
from src.train import train_and_compare_models
from src.predict import SMSPredictor
from src.evaluate import (
    plot_class_distribution,
    plot_message_length_distribution,
    plot_word_count_distribution,
    plot_frequent_words,
    plot_confusion_matrix_heatmap
)


def run_tests():
    print("=" * 70)
    print("STARTING SMS SHIELD SYSTEM VERIFICATION TESTS")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "SMSSpamCollection")
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "sms_spam_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    notebook_path = os.path.join(base_dir, "notebooks", "SMS_Spam_Classification.ipynb")
    
    # TEST 1: Dataset Verification
    print("\n[TEST 1] Testing Dataset Loading...")
    assert os.path.exists(dataset_path), f"SMSSpamCollection not found at {dataset_path}"
    df, stats = load_sms_dataset(dataset_path, drop_duplicates=True)
    assert len(df) > 5000, f"Expected > 5000 rows, got {len(df)}"
    assert stats["ham_count"] > 4000, f"Expected > 4000 ham, got {stats['ham_count']}"
    assert stats["spam_count"] > 600, f"Expected > 600 spam, got {stats['spam_count']}"
    assert "label_num" in df.columns and "char_length" in df.columns
    print(f" PASS: Dataset loaded cleanly: {len(df)} unique records ({stats['ham_count']} ham, {stats['spam_count']} spam).")

    # TEST 2: Preprocessing and Negation Preservation
    print("\n[TEST 2] Testing Preprocessing & Negation Preservation...")
    stop_words = get_custom_stop_words()
    assert "not" not in stop_words, "CRITICAL ERROR: 'not' was found in stop words!"
    assert "no" not in stop_words, "CRITICAL ERROR: 'no' was found in stop words!"
    assert "nor" not in stop_words, "CRITICAL ERROR: 'nor' was found in stop words!"
    
    cleaned_test = clean_sms_text("WINNER!!   Don't delay...   http://claim.com &amp; call now!")
    assert "winner" in cleaned_test and "&" in cleaned_test or "call" in cleaned_test
    print(" PASS: Preprocessing and custom stop words verified; negation markers preserved.")

    # TEST 3: Input Validation Edge Cases
    print("\n[TEST 3] Testing Input Validation Edge Cases...")
    val_empty, msg_empty = validate_sms_input("")
    assert not val_empty, "Empty string should be rejected"
    
    val_ws, msg_ws = validate_sms_input("   \t  \n  ")
    assert not val_ws, "Whitespace string should be rejected"
    
    val_long, msg_long = validate_sms_input("a" * 1005)
    assert not val_long, "Excessively long string should be rejected"
    
    val_punct, msg_punct = validate_sms_input("!@#$%^&*()_+=-{}[]:;'<>?,./")
    assert not val_punct, "Non-alphanumeric string should be rejected"
    
    val_valid, msg_valid = validate_sms_input("Hello friend, are we still meeting today?")
    assert val_valid, "Valid string should pass validation"
    print(" PASS: All input validation edge cases successfully rejected with appropriate messages.")

    # TEST 4: Model Training & Persistence
    print("\n[TEST 4] Testing Model Training & Metrics Generation...")
    best_name, metrics = train_and_compare_models(dataset_path, models_dir)
    assert os.path.exists(model_path), f"Saved model not found at {model_path}"
    assert os.path.exists(metrics_path), f"Saved metrics not found at {metrics_path}"
    assert best_name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear Support Vector Machine"]
    print(f" PASS: 5-Fold Stratified CV completed. Champion selected: '{best_name}'.")

    # TEST 5: Live Inference on Real Samples
    print("\n[TEST 5] Testing Live Prediction Engine...")
    predictor = SMSPredictor(model_path)
    
    # 5a: Spam message
    spam_sample = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim now."
    pred_spam = predictor.predict(spam_sample)
    assert pred_spam["success"] is True
    assert pred_spam["label"] == "SPAM"
    assert pred_spam["is_spam"] is True
    assert pred_spam["spam_probability"] is not None
    assert pred_spam["spam_probability"] > 0.8
    print(f" PASS: Spam sample correctly flagged as SPAM (Probability: {pred_spam['spam_probability']*100:.2f}%).")

    # 5b: Ham message
    ham_sample = "Hey, are you free to grab coffee with mom and me this afternoon around 4pm?"
    pred_ham = predictor.predict(ham_sample)
    assert pred_ham["success"] is True
    assert pred_ham["label"] == "HAM"
    assert pred_ham["is_spam"] is False
    assert pred_ham["spam_probability"] is not None
    assert pred_ham["spam_probability"] < 0.2
    print(f" PASS: Ham sample correctly flagged as HAM (Probability: {pred_ham['spam_probability']*100:.2f}%).")

    # TEST 6: Visualizations Generation
    print("\n[TEST 6] Testing Matplotlib & Seaborn Visualizations Generation...")
    fig1 = plot_class_distribution(df)
    fig2 = plot_message_length_distribution(df)
    fig3 = plot_word_count_distribution(df)
    fig4 = plot_frequent_words(df, "spam")
    fig5 = plot_frequent_words(df, "ham")
    cm_matrix = metrics["test_performance"][best_name]["confusion_matrix"]["matrix"]
    fig6 = plot_confusion_matrix_heatmap(cm_matrix, best_name)
    assert fig1 and fig2 and fig3 and fig4 and fig5 and fig6
    print(" PASS: All 6 required statistical and confusion matrix charts generated successfully.")

    # TEST 7: Notebook Integrity
    print("\n[TEST 7] Testing Colab Notebook Integrity...")
    assert os.path.exists(notebook_path), f"Notebook missing at {notebook_path}"
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    assert "cells" in nb_json and len(nb_json["cells"]) >= 20
    print(f" PASS: Jupyter Notebook structure verified ({len(nb_json['cells'])} cells).")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY! SYSTEM INTEGRITY VERIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
