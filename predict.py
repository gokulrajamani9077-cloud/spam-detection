"""
Root Prediction Interface for SMS Shield
Allows quick standalone prediction from command line or import:
  python predict.py "Your SMS text here"
  python predict.py (interactive mode)
"""

import sys
import os

# Include project directories in path
base_dir = os.path.dirname(os.path.abspath(__file__))
sms_pkg_dir = os.path.join(base_dir, "SMS_SPAM_CLASSIFICATION")
for d in [base_dir, sms_pkg_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from src.predict import SMSPredictor, predict, predict_sms_message, interactive_terminal_mode

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        predictor = SMSPredictor()
        result = predictor.predict(message)
        print("\n" + "=" * 60)
        print(f"INPUT      : {result['input_message']}")
        print(f"PREDICTION : {result['label']}")
        if result.get("spam_probability") is not None:
            print(f"SPAM RISK  : {result['spam_probability'] * 100:.2f}%")
        print(f"EXPLANATION: {result['explanation']}")
        print(f"GUIDANCE   : {result['safety_guidance']}")
        print("=" * 60 + "\n")
    else:
        predictor = SMSPredictor()
        interactive_terminal_mode(predictor)
