"""
SMS Prediction Inference Module
Loads the trained pipeline artifact, executes rigorous input validation,
and generates authentic predictions with contextual explanations.
Can be used as a Python library, CLI command, or interactive terminal tool.
"""

import os
import sys
import argparse
import joblib
from typing import Dict, Any, Optional

# Ensure package paths resolve
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import validate_sms_input, clean_sms_text


class SMSPredictor:
    """Wrapper for trained SMS spam classification pipeline."""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Look in standard locations
            candidate_paths = [
                os.path.join(current_dir, "..", "models", "sms_spam_model.joblib"),
                os.path.join(current_dir, "models", "sms_spam_model.joblib"),
                os.path.join(os.getcwd(), "models", "sms_spam_model.joblib"),
                os.path.join(os.getcwd(), "SMS_SPAM_CLASSIFICATION", "models", "sms_spam_model.joblib"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    model_path = os.path.abspath(p)
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model artifact not found. Please run model training first via 'python -m src.train'."
            )
        
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.classifier_name = type(self.classifier).__name__

    def predict(self, raw_message: str) -> Dict[str, Any]:
        """
        Validates raw SMS message, predicts class, and generates authentic scoring.
        """
        # 1. Input Validation
        is_valid, validation_msg = validate_sms_input(raw_message)
        if not is_valid:
            return {
                "success": False,
                "error": validation_msg,
                "input_message": raw_message,
                "label": "INVALID",
                "is_spam": None,
                "model_name": self.classifier_name,
                "spam_probability": None,
                "score_type": None,
                "explanation": validation_msg,
                "safety_guidance": "Please enter a valid SMS message for analysis."
            }

        # 2. Prediction
        cleaned = clean_sms_text(raw_message)
        pred_label_idx = int(self.pipeline.predict([raw_message])[0])
        is_spam = (pred_label_idx == 1)
        label_text = "SPAM" if is_spam else "HAM"

        # 3. Genuine Probability or Decision Score Handling
        probability = None
        decision_score = None
        score_type = "none"

        if hasattr(self.classifier, "predict_proba"):
            proba_arr = self.pipeline.predict_proba([raw_message])[0]
            probability = float(proba_arr[1])  # Probability of being spam
            score_type = "genuine_probability"
        elif hasattr(self.classifier, "decision_function"):
            decision_score = float(self.pipeline.decision_function([raw_message])[0])
            score_type = "svm_hyperplane_distance"

        # 4. Feature Inspection for Explainability
        words_in_msg = set(cleaned.split())
        vocab = self.vectorizer.vocabulary_
        matched_tokens = [w for w in words_in_msg if w in vocab]

        # 5. Explanatory Context and Safety Guidance
        if is_spam:
            explanation = (
                f"The AI model identified this message as SPAM. It exhibits linguistic patterns "
                f"and vocabulary frequently associated with unsolicited promotions, prizes, "
                f"or fraudulent attempts."
            )
            safety_guidance = (
                "⚠️ Safety Guidance: Do not click unknown links, do not reply with personal "
                "or financial details, and do not call back telephone numbers provided in the message."
            )
        else:
            explanation = (
                f"The AI model categorized this message as HAM (Legitimate). The vocabulary, "
                f"syntax, and tone conform with typical personal or authentic transactional communication."
            )
            safety_guidance = (
                "✅ Safety Guidance: This message appears benign. Exercise standard caution "
                "whenever interacting with unexpected requests."
            )

        return {
            "success": True,
            "error": None,
            "input_message": raw_message,
            "cleaned_message": cleaned,
            "label": label_text,
            "is_spam": is_spam,
            "model_name": self.classifier_name,
            "spam_probability": probability,
            "decision_score": decision_score,
            "score_type": score_type,
            "matched_features_count": len(matched_tokens),
            "explanation": explanation,
            "safety_guidance": safety_guidance
        }


def predict_sms_message(raw_message: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience helper for standalone prediction."""
    predictor = SMSPredictor(model_path=model_path)
    return predictor.predict(raw_message)


# Short alias for quick imports
predict = predict_sms_message


def interactive_terminal_mode(predictor: SMSPredictor):
    """Interactive CLI terminal session for live SMS testing."""
    print("=" * 70)
    print("      SMS SHIELD — INTERACTIVE LIVE PREDICTION TERMINAL")
    print(f"      Active Model: {predictor.classifier_name}")
    print("=" * 70)
    print("Type any SMS message below and press Enter to see prediction.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("SMS Input > ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting SMS Shield. Stay secure!")
                break
            if not user_input:
                continue

            result = predictor.predict(user_input)
            print("-" * 70)
            if not result["success"]:
                print(f"[REJECTED] {result['error']}")
            else:
                label_color = "SPAM [MALICIOUS / UNSOLICITED]" if result["is_spam"] else "HAM [LEGITIMATE / BENIGN]"
                print(f"PREDICTION : {label_color}")
                if result["spam_probability"] is not None:
                    print(f"SPAM RISK  : {result['spam_probability'] * 100:.2f}% probability")
                elif result["decision_score"] is not None:
                    print(f"SVM SCORE  : {result['decision_score']:.3f} (hyperplane distance)")
                print(f"EXPLANATION: {result['explanation']}")
                print(f"GUIDANCE   : {result['safety_guidance']}")
            print("-" * 70 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMS Shield Spam Classification CLI")
    parser.add_argument("message", nargs="?", type=str, help="SMS text to classify (optional)")
    args = parser.parse_args()

    pred_engine = SMSPredictor()
    if args.message:
        res = pred_engine.predict(args.message)
        print("\n--- SMS CLASSIFICATION RESULT ---")
        print(f"Input       : {res['input_message']}")
        print(f"Prediction  : {res['label']}")
        if res.get("spam_probability") is not None:
            print(f"Spam Risk   : {res['spam_probability'] * 100:.2f}%")
        print(f"Explanation : {res['explanation']}")
        print(f"Guidance    : {res['safety_guidance']}\n")
    else:
        interactive_terminal_mode(pred_engine)
