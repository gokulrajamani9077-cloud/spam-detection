"""
SMS SHIELD — AI-POWERED SPAM DETECTION SYSTEM
Streamlit Web Application
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="SMS Shield | AI Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve import paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sms_pkg_dir = os.path.join(CURRENT_DIR, "SMS_SPAM_CLASSIFICATION")
for p in [CURRENT_DIR, sms_pkg_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_sms_dataset
from src.preprocessing import validate_sms_input, clean_sms_text
from src.predict import SMSPredictor
from src.train import train_and_compare_models
from src.evaluate import (
    plot_class_distribution,
    plot_message_length_distribution,
    plot_word_count_distribution,
    plot_frequent_words,
    plot_confusion_matrix_heatmap,
    explain_confusion_matrix
)

# Robust Path Resolution
def resolve_file(rel_path: str) -> str:
    candidates = [
        os.path.join(CURRENT_DIR, rel_path),
        os.path.join(CURRENT_DIR, "SMS_SPAM_CLASSIFICATION", rel_path),
        os.path.join(os.getcwd(), rel_path),
        os.path.join(os.getcwd(), "SMS_SPAM_CLASSIFICATION", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(CURRENT_DIR, rel_path)


DATASET_PATH = resolve_file(os.path.join("dataset", "SMSSpamCollection"))
MODELS_DIR = os.path.dirname(resolve_file(os.path.join("models", "sms_spam_model.joblib")))
MODEL_FILE = resolve_file(os.path.join("models", "sms_spam_model.joblib"))
METRICS_FILE = resolve_file(os.path.join("models", "metrics.json"))

# Custom UI Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.8rem;
    }
    
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.25rem;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    .result-box-spam {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.6rem;
        margin: 1.2rem 0;
    }
    
    .result-box-ham {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        border-radius: 14px;
        padding: 1.6rem;
        margin: 1.2rem 0;
    }
    
    .guidance-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #334155;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "sms_input_box" not in st.session_state:
    st.session_state.sms_input_box = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim now. Code KL341."
if "current_result" not in st.session_state:
    st.session_state.current_result = None


@st.cache_data
def get_dataset():
    """Loads and caches the SMS dataset."""
    df, stats = load_sms_dataset(DATASET_PATH, drop_duplicates=True)
    return df, stats


@st.cache_data
def get_metrics():
    """Loads saved model evaluation metrics."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def get_predictor():
    """Loads and caches the machine learning predictor."""
    candidate_paths = [
        MODEL_FILE,
        os.path.join(CURRENT_DIR, "models", "sms_spam_model.joblib"),
        os.path.join(CURRENT_DIR, "SMS_SPAM_CLASSIFICATION", "models", "sms_spam_model.joblib"),
        os.path.join(os.getcwd(), "models", "sms_spam_model.joblib"),
        os.path.join(os.getcwd(), "SMS_SPAM_CLASSIFICATION", "models", "sms_spam_model.joblib"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return SMSPredictor(p)
    return None


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🛡️ **SMS SHIELD AI**")
    st.markdown("Academic Spam Detection System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "📊 Dataset Explorer",
            "⚙️ Model Training",
            "📈 Model Evaluation",
            "🔍 Live SMS Prediction",
            "ℹ️ About Project"
        ]
    )
    
    st.markdown("---")
    st.markdown("**Dataset Source:**")
    st.caption("UCI Machine Learning Repository: SMS Spam Collection Corpus")
    st.caption("Almeida et al. (2011)")
    
    st.markdown("---")
    st.caption("Built with Scikit-Learn & Streamlit")


# PAGE 1: HOME DASHBOARD
if page == "🏠 Home Dashboard":
    st.markdown('<h1 class="main-title">SMS SHIELD — AI SPAM DETECTION SYSTEM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Production-grade natural language processing system for real-time mobile message security.</p>', unsafe_allow_html=True)
    
    df, stats = get_dataset()
    metrics = get_metrics()
    
    best_model_name = metrics.get("best_model", "Multinomial Naive Bayes") if metrics else "Multinomial Naive Bayes"
    best_summary = metrics.get("best_model_test_summary", {}) if metrics else {}
    test_acc = best_summary.get("accuracy", 0.9835) * 100
    test_f1 = best_summary.get("f1", 0.9306) * 100
    test_prec = best_summary.get("precision", 0.9744) * 100
    test_rec = best_summary.get("recall", 0.8906) * 100

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Clean Messages</div>
            <div class="metric-value">{stats['clean_total']:,}</div>
            <div class="metric-sub">{stats['total_raw']:,} raw collected</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ham (Legitimate)</div>
            <div class="metric-value">{stats['ham_count']:,}</div>
            <div class="metric-sub">{100 - stats['spam_percentage']:.1f}% of corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Spam Intercepted</div>
            <div class="metric-value">{stats['spam_count']:,}</div>
            <div class="metric-sub">{stats['spam_percentage']:.1f}% spam ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test F1-Score</div>
            <div class="metric-value">{test_f1:.2f}%</div>
            <div class="metric-sub">{test_acc:.2f}% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.6, 1])
    with col_left:
        st.markdown("### 🎯 System Overview")
        st.write(
            """
            **SMS Shield** is an end-to-end artificial intelligence architecture engineered to distinguish 
            legitimate SMS messages (**Ham**) from fraudulent, phishing, and promotional communications (**Spam**).
            
            - **End-to-End Pipeline**: Combines negation-preserving linguistic normalization, n-gram TF-IDF vectorization, and probabilistic classification.
            - **Preventing Data Leakage**: Vectorization occurs strictly inside Scikit-learn pipelines with 5-Fold Stratified Cross-Validation.
            - **Academic Grounding**: Evaluated rigorously using True/False Positive and Negative rates to balance security against false alarm inconvenience.
            """
        )
        st.info(f"🏆 **Current Champion Model**: **{best_model_name}** selected based on highest 5-fold cross-validation F1-Score.")

    with col_right:
        st.markdown("### ⚡ Live Model Status")
        st.markdown(f"""
        - **Vectorization**: TF-IDF (Unigrams & Bigrams)
        - **Vocabulary Size**: 5,000 top features
        - **Cross-Validation**: 5-Fold Stratified K-Fold
        - **Test Set**: 20% Held-Out ({metrics.get('test_samples', 1032)} messages)
        - **Precision**: {test_prec:.2f}% (Minimizes false alarms)
        - **Recall**: {test_rec:.2f}% (Interception rate)
        """)

    st.markdown("---")
    st.markdown("### 📋 Recent Session Predictions")
    if len(st.session_state.prediction_history) == 0:
        st.caption("No live predictions performed in this session yet. Navigate to 'Live SMS Prediction' to test messages.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(
            history_df[["timestamp", "label", "spam_probability", "message"]],
            use_container_width=True,
            hide_index=True
        )


# PAGE 2: DATASET EXPLORER
elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="main-title">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Exploratory Data Analysis and statistical inspection of the UCI SMS Spam Collection.</p>', unsafe_allow_html=True)
    
    df, stats = get_dataset()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Messages", f"{stats['total_raw']:,}")
    m2.metric("Duplicates Handled", f"{stats['duplicate_count']:,}")
    m3.metric("Unique Clean Messages", f"{stats['clean_total']:,}")
    m4.metric("Missing Values", f"{stats['missing_count']}")
    
    st.markdown("---")
    st.markdown("### 🔎 Dataset Sample Browser")
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        class_filter = st.selectbox("Filter by Category", ["All", "ham", "spam"])
    with f_col2:
        search_kw = st.text_input("Search messages by keyword", "")
        
    filtered_df = df.copy()
    if class_filter != "All":
        filtered_df = filtered_df[filtered_df["label"] == class_filter]
    if search_kw.strip():
        filtered_df = filtered_df[filtered_df["message"].str.contains(search_kw.strip(), case=False, na=False)]
        
    st.dataframe(
        filtered_df[["label", "message", "char_length", "word_count"]].head(100),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Displaying top matching entries ({len(filtered_df):,} total matches found).")
    
    st.markdown("---")
    st.markdown("### 📈 Analytical Distributions")
    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Message Length Analysis", "Top Frequent Lexicons"])
    
    with tab1:
        col_plot, col_expl = st.columns([1.5, 1])
        with col_plot:
            fig1 = plot_class_distribution(df)
            st.pyplot(fig1)
        with col_expl:
            st.markdown("#### Class Imbalance Context")
            st.write(
                f"""
                - **Ham Messages**: {stats['ham_count']:,} ({100 - stats['spam_percentage']:.1f}%)
                - **Spam Messages**: {stats['spam_count']:,} ({stats['spam_percentage']:.1f}%)
                
                **Academic Implication**:
                The dataset displays typical real-world class imbalance (~87.5% legitimate vs ~12.5% spam).
                Standard accuracy can be deceptive in imbalanced environments.
                Therefore, **F1-score** and **Precision-Recall metrics** are prioritized.
                """
            )
            
    with tab2:
        c_len, c_words = st.columns(2)
        with c_len:
            fig2 = plot_message_length_distribution(df)
            st.pyplot(fig2)
            st.caption(f"Average Ham Length: {stats['avg_ham_length']} chars | Average Spam Length: {stats['avg_spam_length']} chars")
        with c_words:
            fig3 = plot_word_count_distribution(df)
            st.pyplot(fig3)
            st.caption(f"Average Ham Words: {stats['avg_ham_words']} words | Average Spam Words: {stats['avg_spam_words']} words")
            
    with tab3:
        col_spam_words, col_ham_words = st.columns(2)
        with col_spam_words:
            fig_spam = plot_frequent_words(df, "spam")
            st.pyplot(fig_spam)
        with col_ham_words:
            fig_ham = plot_frequent_words(df, "ham")
            st.pyplot(fig_ham)


# PAGE 3: MODEL TRAINING
elif page == "⚙️ Model Training":
    st.markdown('<h1 class="main-title">⚙️ Machine Learning Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Algorithm comparison, 5-Fold Stratified Cross-Validation, and pipeline compilation.</p>', unsafe_allow_html=True)
    
    metrics = get_metrics()
    
    st.markdown("### 🔬 Candidate Algorithms")
    st.write(
        """
        1. **Multinomial Naive Bayes**: A probabilistic baseline rooted in Bayes' Theorem, exceptionally effective on text.
        2. **Logistic Regression**: A generalized linear model using a sigmoid function to predict calibrated probabilities.
        3. **Linear Support Vector Machine (LinearSVC)**: A margin-maximizing hyperplane separator.
        """
    )
    st.markdown("---")
    
    if metrics:
        st.markdown("### 📊 5-Fold Stratified Cross-Validation Results")
        cv_res = metrics.get("cross_validation_results", {})
        
        cv_rows = []
        for name, scores in cv_res.items():
            cv_rows.append({
                "Algorithm": name,
                "CV F1-Score": f"{scores['cv_f1_mean'] * 100:.2f}% ± {scores['cv_f1_std'] * 100:.2f}%",
                "CV Accuracy": f"{scores['cv_accuracy_mean'] * 100:.2f}% ± {scores['cv_accuracy_std'] * 100:.2f}%",
                "CV Precision": f"{scores['cv_precision_mean'] * 100:.2f}% ± {scores['cv_precision_std'] * 100:.2f}%",
                "CV Recall": f"{scores['cv_recall_mean'] * 100:.2f}% ± {scores['cv_recall_std'] * 100:.2f}%",
                "Train F1": f"{scores['train_f1_mean'] * 100:.2f}%"
            })
            
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
        
        best_name = metrics.get("best_model", "Multinomial Naive Bayes")
        st.success(
            f"🏆 **Selected Model**: **{best_name}** achieved the superior 5-Fold Cross-Validation F1-Score "
            f"({cv_res[best_name]['cv_f1_mean'] * 100:.2f}%)."
        )
        
    st.markdown("---")
    st.markdown("### 🔄 Retrain Pipeline Live")
    if st.button("🚀 Retrain Model Now", type="primary"):
        with st.spinner("Training models across 5 folds and compiling pipeline..."):
            best_model_name, new_metrics = train_and_compare_models(DATASET_PATH, MODELS_DIR)
            time.sleep(1)
        st.success(f"Training completed! Champion model: {best_model_name}")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# PAGE 4: MODEL EVALUATION
elif page == "📈 Model Evaluation":
    st.markdown('<h1 class="main-title">📈 Rigorous Model Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Performance analysis on held-out 20% test partition (1,032 SMS messages).</p>', unsafe_allow_html=True)
    
    metrics = get_metrics()
    if not metrics:
        st.warning("Metrics file not detected.")
    else:
        best_name = metrics["best_model"]
        test_summary = metrics["test_performance"][best_name]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy", f"{test_summary['accuracy'] * 100:.2f}%")
        col2.metric("Test Precision (Spam)", f"{test_summary['precision'] * 100:.2f}%")
        col3.metric("Test Recall (Spam)", f"{test_summary['recall'] * 100:.2f}%")
        col4.metric("Test F1-Score", f"{test_summary['f1'] * 100:.2f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        c_left, c_right = st.columns([1.2, 1.3])
        with c_left:
            cm_matrix = test_summary["confusion_matrix"]["matrix"]
            fig_cm = plot_confusion_matrix_heatmap(cm_matrix, best_name)
            st.pyplot(fig_cm)
            
        with c_right:
            st.markdown("### 🎓 Academic Confusion Matrix Breakdown")
            explanations = explain_confusion_matrix(test_summary["confusion_matrix"])
            for key, desc in explanations.items():
                if "True Negative" in key:
                    st.success(f"**{key}**: {desc}")
                elif "True Positive" in key:
                    st.info(f"**{key}**: {desc}")
                elif "False Positive" in key:
                    st.warning(f"**{key}**: {desc}")
                else:
                    st.error(f"**{key}**: {desc}")
                    
        st.markdown("---")
        st.markdown("### 📋 Full Classification Report")
        clf_rep = test_summary["classification_report"]
        rep_rows = []
        for label_key in ["ham", "spam"]:
            if label_key in clf_rep:
                rep_rows.append({
                    "Class": label_key.upper(),
                    "Precision": f"{clf_rep[label_key]['precision'] * 100:.2f}%",
                    "Recall": f"{clf_rep[label_key]['recall'] * 100:.2f}%",
                    "F1-Score": f"{clf_rep[label_key]['f1-score'] * 100:.2f}%",
                    "Support": int(clf_rep[label_key]['support'])
                })
        for avg_key in ["macro avg", "weighted avg"]:
            if avg_key in clf_rep:
                rep_rows.append({
                    "Class": avg_key.title(),
                    "Precision": f"{clf_rep[avg_key]['precision'] * 100:.2f}%",
                    "Recall": f"{clf_rep[avg_key]['recall'] * 100:.2f}%",
                    "F1-Score": f"{clf_rep[avg_key]['f1-score'] * 100:.2f}%",
                    "Support": int(clf_rep[avg_key]['support'])
                })
        st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)


# PAGE 5: LIVE SMS PREDICTION (ENHANCED & BULLETPROOF)
elif page == "🔍 Live SMS Prediction":
    st.markdown('<h1 class="main-title">🔍 Live SMS Spam Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Test any SMS message against the production machine learning pipeline.</p>', unsafe_allow_html=True)
    
    predictor = get_predictor()
    if not predictor:
        st.error("Trained model artifact not found. Please train the model via the Model Training tab.")
    else:
        # Quick Sample Buttons
        st.markdown("#### ⚡ Quick Samples (Click any to load and analyze instantly):")
        b1, b2, b3, b4 = st.columns(4)
        
        sample_selected = None
        if b1.button("📩 Spam: Prize Winner", use_container_width=True):
            sample_selected = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim your code KL341 now. Valid 12 hours only."
        if b2.button("📩 Spam: Urgent Claim", use_container_width=True):
            sample_selected = "URGENT! Your mobile number won 2000 bonus cash. Call 08718729758 immediately. 2nd attempt to reach you!"
        if b3.button("💬 Ham: Meeting Dinner", use_container_width=True):
            sample_selected = "Hey! Are you still meeting us at the cafe for dinner tonight? Let me know when you leave work."
        if b4.button("💬 Ham: Family Chat", use_container_width=True):
            sample_selected = "Hi dad, I am on the way home now and will pick up groceries. See you soon!"

        # Handle sample button clicks: updates session state and calculates immediately
        if sample_selected:
            st.session_state.sms_input_box = sample_selected
            res_sample = predictor.predict(sample_selected)
            st.session_state.current_result = res_sample
            st.session_state.prediction_history.insert(0, {
                "timestamp": time.strftime("%H:%M:%S"),
                "message": sample_selected[:60] + "...",
                "label": res_sample["label"],
                "spam_probability": f"{res_sample['spam_probability'] * 100:.2f}%" if res_sample.get('spam_probability') is not None else "N/A"
            })
            st.rerun()

        # Text Input Area directly bound to session_state
        user_input = st.text_area(
            "Enter or paste SMS Message Content:",
            key="sms_input_box",
            height=130,
            placeholder="Type or paste any SMS message here...",
            help="Type an SMS message to evaluate with the machine learning model."
        )

        char_len = len(user_input)
        word_count = len(user_input.split())
        st.caption(f"Length: {char_len} characters | Words: {word_count} | Limit: 1,000 characters")

        # Action Buttons
        col_btn1, col_btn2, _ = st.columns([1.5, 1, 3])
        with col_btn1:
            run_clicked = st.button("🛡️ Analyse Message", type="primary", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("🗑️ Clear Input", use_container_width=True)

        if clear_clicked:
            st.session_state.sms_input_box = ""
            st.session_state.current_result = None
            st.rerun()

        if run_clicked:
            if not user_input.strip():
                st.warning("⚠️ Please enter an SMS message before analyzing.")
            else:
                with st.spinner("Classifying message patterns..."):
                    res = predictor.predict(user_input)
                    st.session_state.current_result = res
                    if res["success"]:
                        st.session_state.prediction_history.insert(0, {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "message": user_input[:60] + "..." if len(user_input) > 60 else user_input,
                            "label": res["label"],
                            "spam_probability": f"{res['spam_probability'] * 100:.2f}%" if res.get('spam_probability') is not None else "N/A"
                        })

        # Display Live Result Card
        if st.session_state.current_result is not None:
            res = st.session_state.current_result
            st.markdown("---")
            st.markdown("### 🎯 Classification Result")

            if not res["success"]:
                st.error(f"❌ Input Rejected: {res['error']}")
            else:
                is_spam = res["is_spam"]

                if is_spam:
                    st.markdown(f"""
                    <div class="result-box-spam">
                        <h2 style="color: #9B1C1C; margin: 0 0 0.5rem 0;">🚨 SPAM DETECTED</h2>
                        <p style="font-size: 1.15rem; color: #771D1D; margin: 0; font-weight: 600;">
                            This message is classified as <strong>SPAM / FRAUDULENT</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box-ham">
                        <h2 style="color: #03543F; margin: 0 0 0.5rem 0;">✅ LEGITIMATE MESSAGE (HAM)</h2>
                        <p style="font-size: 1.15rem; color: #046C4E; margin: 0; font-weight: 600;">
                            This message is classified as <strong>HAM (LEGITIMATE)</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics row
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    st.metric("Model Used", res["model_name"])
                with m_c2:
                    if res["spam_probability"] is not None:
                        st.metric("Spam Risk Score", f"{res['spam_probability'] * 100:.2f}%")
                    elif res["decision_score"] is not None:
                        st.metric("SVM Hyperplane Margin", f"{res['decision_score']:.3f}")
                with m_c3:
                    st.metric("Salient Words Matched", res["matched_features_count"])

                # Spam Risk Gauge
                if res["spam_probability"] is not None:
                    p_val = float(res["spam_probability"])
                    st.markdown("**Spam Probability Gauge:**")
                    st.progress(p_val)
                    if p_val > 0.8:
                        st.caption("🔴 High Confidence Spam Pattern")
                    elif p_val > 0.5:
                        st.caption("🟠 Moderate Spam Risk")
                    else:
                        st.caption("🟢 Safe Benign Communication")

                # Explanation and Guidance
                st.markdown("#### 🧠 AI Explanation")
                st.write(res["explanation"])

                st.markdown(f"""
                <div class="guidance-box">
                    {res['safety_guidance']}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Recent Live Predictions in This Session")
        if len(st.session_state.prediction_history) == 0:
            st.caption("No prediction history recorded yet.")
        else:
            h_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(h_df[["timestamp", "label", "spam_probability", "message"]], use_container_width=True, hide_index=True)


# PAGE 6: ABOUT PROJECT
elif page == "ℹ️ About Project":
    st.markdown('<h1 class="main-title">ℹ️ About SMS Shield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">System documentation, methodology, academic attribution, and viva preparation.</p>', unsafe_allow_html=True)
    
    tab_doc, tab_cite, tab_viva = st.tabs(["System Architecture", "Dataset Attribution", "Academic Viva Q&A"])
    
    with tab_doc:
        st.markdown(
            """
            ### 📌 Project Objective
            To engineer an accurate, leak-free, production-ready machine learning system capable of intercepting 
            fraudulent SMS messages with high precision (avoiding dropping critical personal messages) while maintaining high recall.
            
            ### 🛠️ Technology Stack
            - **Python 3.12**: Core programming language.
            - **Scikit-Learn**: Machine learning pipelines, TF-IDF vectorization, Stratified K-Fold CV, and evaluation.
            - **Streamlit**: Responsive interactive web application and dashboard.
            - **Pandas & NumPy**: Data ingestion, manipulation, and numerical operations.
            - **Matplotlib & Seaborn**: Statistical distribution visualizations.
            - **Joblib**: Model serialization and persistence.
            """
        )
        
    with tab_cite:
        st.markdown(
            """
            ### 📚 Dataset Attribution: UCI Machine Learning Repository
            This project employs the authentic **SMS Spam Collection v.1** compiled by Tiago Agostinho de Almeida and José María Gómez Hidalgo.
            
            **Citation**:
            > Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A.  
            > *Contributions to the study of SMS Spam Filtering: New Collection and Results.*  
            > Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11), Mountain View, CA, USA, 2011.
            """
        )
        
    with tab_viva:
        st.markdown(
            """
            ### 🎓 Academic Viva Questions & Answers
            
            **Q1: Why is Accuracy an insufficient metric for SMS spam filtering?**  
            *Answer*: Because SMS datasets have class imbalance (~87.5% ham vs 12.5% spam). A dumb model predicting 'ham' on every input achieves 87.5% accuracy but catches 0% spam. F1-score balances precision and recall on the minority spam class.
            
            **Q2: What is the danger of removing negation words in text preprocessing?**  
            *Answer*: Traditional stop word lists remove words like "not" or "no". In an SMS, "Call me, I will not pay" vs "Call me, I will pay" have diametrically opposite meanings. Preserving negations retains essential semantic polarity.
            
            **Q3: How was data leakage prevented in this project?**  
            *Answer*: Feature extraction (TF-IDF) was placed strictly inside a Scikit-learn `Pipeline`. During 5-fold cross-validation, the vectorizer fits exclusively on the 4 training folds and transforms the validation fold, preventing vocabulary leakage.
            """
        )
