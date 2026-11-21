# 🛡️ SMS SHIELD — AI-POWERED SMS SPAM CLASSIFICATION SYSTEM

An academic, production-grade Natural Language Processing (NLP) system engineered to accurately intercept mobile SMS spam, phishing scams, and marketing broadcasts while preserving legitimate personal communications.

---

## 📌 Project Overview

SMS messaging remains a mission-critical communication channel with an estimated 98% open rate. Consequently, it is increasingly targeted by automated spam and smishing campaigns. **SMS Shield** provides an end-to-end, leak-free machine learning architecture that:
- Ingests the original **UCI SMS Spam Collection** corpus.
- Employs **negation-preserving linguistic normalization** to retain semantic meaning.
- Uses **TF-IDF n-gram vectorization** within Scikit-Learn pipelines to prevent data leakage.
- Validates candidate algorithms via **5-Fold Stratified Cross-Validation**.
- Features an interactive **Streamlit dashboard** for real-time inference, model exploration, and analytics.

---

## 🏛️ System Architecture

```text
SMS_SPAM_CLASSIFICATION/
│
├── dataset/
│   ├── SMSSpamCollection          # Authentic UCI tab-separated corpus
│   └── readme                     # Original UCI dataset documentation
│
├── notebooks/
│   └── SMS_Spam_Classification.ipynb  # End-to-end Google Colab / Jupyter notebook
│
├── models/
│   ├── sms_spam_model.joblib      # Serialized Scikit-learn champion pipeline
│   └── metrics.json               # Full cross-validation and test evaluation records
│
├── src/
│   ├── data_loader.py             # Dataset ingestion and exploratory statistics
│   ├── preprocessing.py           # Negation-preserving NLP and input validation
│   ├── train.py                   # 5-fold CV, comparison, and model persistence
│   ├── evaluate.py                # Visualizations and confusion matrix explanations
│   └── predict.py                 # Production inference and explainability engine
│
├── app.py                         # Professional multi-page Streamlit web application
├── requirements.txt               # Locked, verified dependencies
├── README.md                      # Comprehensive technical documentation
├── .gitignore                     # Git configuration
└── LICENSE                        # Open-source license
```

---

## 📊 Dataset Attribution & Analysis

- **Source**: UCI Machine Learning Repository — *SMS Spam Collection v.1*
- **Curators**: Tiago Agostinho de Almeida and José María Gómez Hidalgo
- **Reference**:
  > Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A.  
  > *Contributions to the study of SMS Spam Filtering: New Collection and Results.*  
  > Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11), Mountain View, CA, USA, 2011.

### Dataset Key Figures:
- **Raw Messages**: 5,574 messages (tab-separated, no headers).
- **Duplicate Handling**: 414 duplicate SMS messages identified and removed to prevent test-set contamination.
- **Clean Unique Corpus**: 5,160 messages.
  - **Ham (Legitimate)**: 4,518 (87.56%)
  - **Spam**: 642 (12.44%)
- **Average Character Length**: Ham: 70.9 characters | Spam: 137.4 characters.
- **Average Word Count**: Ham: 14.2 words | Spam: 23.7 words.

---

## 🔬 Machine Learning Methodology

### 1. Negation-Preserving Preprocessing
Standard stop-word lists discard terms like `"not"`, `"no"`, and `"never"`. In SMS text, removing negations completely reverses meaning (e.g., *"not subscribed"* vs *"subscribed"*). SMS Shield selectively retains all negation operators.

### 2. Preventing Data Leakage
Feature extraction (TF-IDF) is strictly encapsulated inside `sklearn.pipeline.Pipeline`. In 5-fold cross-validation, the vectorizer fits exclusively on training folds and merely transforms validation folds, guaranteeing zero data leakage.

### 3. Model Benchmark
Three supervised algorithms were compared under 5-Fold Stratified Cross-Validation on the 80% training split:
1. **Multinomial Naive Bayes (`alpha=0.1`)**
2. **Logistic Regression (`C=1.0`)**
3. **Linear Support Vector Machine (`LinearSVC`)**

---

## 📈 Evaluation & Results

### 5-Fold Stratified Cross-Validation Performance (Training Split):
| Algorithm | CV F1-Score | CV Accuracy | CV Precision | CV Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** 🏆 | **92.58% ± 1.47%** | **98.26% ± 0.32%** | **98.06% ± 0.76%** | 87.75% ± 2.97% |
| **Linear Support Vector Machine** | 92.55% ± 2.11% | 98.23% ± 0.49% | 97.22% ± 1.99% | **88.33% ± 2.66%** |
| **Logistic Regression** | 76.89% ± 5.37% | 95.28% ± 0.95% | 97.29% ± 2.48% | 63.83% ± 6.95% |

### Champion Model Evaluation on Held-Out Test Set (1,032 Messages):
- **Accuracy**: `98.35%`
- **Precision (Spam)**: `97.44%`
- **Recall (Spam)**: `89.06%`
- **F1-Score**: `93.06%`
- **ROC-AUC**: `99.16%`

### Confusion Matrix Breakdown:
- **True Negative (TN = 901)**: Legitimate messages received unhindered.
- **False Positive (FP = 3)**: Legitimate messages misclassified as spam (Type I Error). Extremely low rate preserves user trust.
- **False Negative (FN = 14)**: Spam messages that bypassed filtering (Type II Error).
- **True Positive (TP = 114)**: Spam messages accurately flagged and quarantined.

---

## 💻 Local Setup & Execution

### 1. Clone or Extract the Project
```bash
git clone https://github.com/<your-username>/sms-spam-classification.git
cd sms-spam-classification
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional — Pre-trained Model Included)
```bash
python -m src.train
```

### 5. Launch the Streamlit Web Application
```bash
start.bat    # Or: streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment Guide

Follow these exact steps to deploy SMS Shield as a public web application:

1. **Push the Project to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete SMS Shield spam classification project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sms-spam-classification.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3. **Deploy the App**:
   - Click the **"New app"** button.
   - Select your repository: `<your-username>/sms-spam-classification`.
   - Set **Branch**: `main`.
   - Set **Main file path**: `app.py`.
   - Click **"Deploy!"**.

4. **Public Access**:
   - Streamlit Cloud builds the environment from `requirements.txt`.
   - Your public application will be available at `https://<your-app-name>.streamlit.app`.

---

## 🎓 Academic Viva Questions & Answers

**Q1: Why is Accuracy misleading for evaluating spam filters?**  
*Answer*: Because of class imbalance (~87.5% ham vs 12.5% spam). A baseline model predicting "ham" on every message would score 87.5% accuracy while failing 100% of spam detections. The F1-score balances precision and recall on the minority spam class.

**Q2: What is the risk of removing negation words in text preprocessing?**  
*Answer*: Traditional stop word lists remove words like "not" or "no". In an SMS, "I will not pay" vs "I will pay" have opposite meanings. Retaining negations preserves sentiment and intent.

**Q3: How was data leakage prevented in this system?**  
*Answer*: TF-IDF feature extraction was placed inside a Scikit-learn Pipeline. During cross-validation, the vectorizer was fitted exclusively on the training folds and only transformed the validation fold.

---

## 📄 License
This project is released under the MIT Open Source License.
Attribution remains with the UCI Machine Learning Repository and the original dataset authors.
