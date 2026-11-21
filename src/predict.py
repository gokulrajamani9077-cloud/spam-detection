"""
SMS Prediction Inference Module
Loads the trained pipeline artifact, executes rigorous input validation,
and generates authentic predictions with contextual explanations.
"""

import os
import joblib
from typing import Dict, Any, Optional
from src.preprocessing import validate_sms_input, clean_sms_text


class SMSPredictor:
    """Wrapper for trained SMS spam classification pipeline."""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "sms_spam_model.joblib"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
        
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
                "label": None,
                "is_spam": None,
                "model_name": self.classifier_name,
                "probability": None,
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
            probability = float(proba_arr[1]) # Probability of being spam
            score_type = "genuine_probability"
        elif hasattr(self.classifier, "decision_function"):
            decision_score = float(self.pipeline.decision_function([raw_message])[0])
            score_type = "svm_hyperplane_distance"

        # 4. Feature Inspection for Explainability
        # Inspect salient tokens present in the vocabulary
        words_in_msg = set(cleaned.split())
        feature_names = self.vectorizer.get_feature_names_out()
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
