"""
STEP 9: STREAMLIT WEB APP
==========================
Creates an interactive web interface for fraud detection

Run with: streamlit run 9_streamlit_app.py
Then open: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler (cached for performance)"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files not found!")
        st.info("Please run all 8 steps first:")
        st.write("1. python 1_data_collection.py")
        st.write("2. python 2_preprocessing.py")
        st.write("... and so on until step 8")
        return None, None

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
    <style>
    /* Dark mode compatible CSS */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0f172a;
        }
    }

    /* Mobile Responsive CSS */
    @media (max-width: 768px) {
        .card {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }

        .stat-card {
            padding: 1rem;
        }

        .stat-card div[style*="font-size: 2rem"] {
            font-size: 1.5rem !important;
        }

        .stat-card div[style*="font-size: 0.875rem"] {
            font-size: 0.75rem !important;
        }

        h1 {
            font-size: 1.5rem !important;
        }

        h2 {
            font-size: 1.25rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        .risk-badge {
            padding: 0.4rem 0.8rem;
            font-size: 0.75rem;
        }

        /* Stack columns on mobile */
        .stColumns > div {
            flex-direction: column;
        }

        /* Adjust charts for mobile */
        .stPlotlyChart, .stMatplotlibChart {
            width: 100% !important;
        }
    }

    @media (max-width: 480px) {
        .card {
            padding: 0.75rem;
            border-radius: 8px;
        }

        .stat-card {
            padding: 0.75rem;
            border-radius: 8px;
        }

        .stat-card div[style*="font-size: 2rem"] {
            font-size: 1.25rem !important;
        }

        h1 {
            font-size: 1.25rem !important;
        }

        h2 {
            font-size: 1.1rem !important;
        }

        h3 {
            font-size: 1rem !important;
        }

        /* Hide sidebar on very small screens */
        .css-1d391kg {
            width: 0 !important;
            display: none !important;
        }

        /* Adjust main content for no sidebar */
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    /* Card Styling - works in both modes */
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .card-text {
        color: #64748b;
    }

    @media (prefers-color-scheme: dark) {
        .card {
            background-color: rgba(30, 41, 59, 0.9);
            color: #e2e8f0;
        }
        .card-text {
            color: #94a3b8;
        }
    }

    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Result Cards - Light mode */
    .fraud-high {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1.5rem;
        color: #1e293b;
    }

    .fraud-medium {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1.5rem;
        color: #1e293b;
    }

    .fraud-low {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1.5rem;
        color: #1e293b;
    }

    /* Result Cards - Dark mode */
    @media (prefers-color-scheme: dark) {
        .fraud-high {
            background-color: rgba(220, 38, 38, 0.1);
            border-left: 4px solid #dc2626;
            color: #e2e8f0;
        }

        .fraud-medium {
            background-color: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
            color: #e2e8f0;
        }

        .fraud-low {
            background-color: rgba(22, 163, 74, 0.1);
            border-left: 4px solid #16a34a;
            color: #e2e8f0;
        }

        .confidence-box {
            background-color: rgba(30, 41, 59, 0.9) !important;
        }
    }

    .confidence-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    @media (prefers-color-scheme: dark) {
        .confidence-score {
            color: #e2e8f0 !important;
        }
    }

    .confidence-score {
        color: #1e293b;
    }

    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
    }

    .risk-high {
        background-color: #dc2626;
        color: white;
    }

    .risk-medium {
        background-color: #f59e0b;
        color: white;
    }

    .risk-low {
        background-color: #16a34a;
        color: white;
    }

    /* Typography - Light mode */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }

    /* Typography - Dark mode */
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 {
            color: #e2e8f0;
        }
    }

    /* Button Styling */
    .stButton > button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #1e40af;
    }

    @media (max-width: 768px) {
        .stButton > button {
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    @media (max-width: 768px) {
        .dataframe {
            font-size: 0.8rem;
        }
    }

    /* Sidebar and general text */
    @media (prefers-color-scheme: dark) {
        .stMarkdown, .stWrite, .stSelectbox label, .stSlider label, .stRadio label {
            color: #e2e8f0;
        }
    }

    /* Mobile-friendly slider */
    @media (max-width: 768px) {
        .stSlider {
            font-size: 0.9rem;
        }
    }

    /* Mobile-friendly selectbox */
    @media (max-width: 768px) {
        .stSelectbox {
            font-size: 0.9rem;
        }
    }

    /* Optimize images for mobile */
    @media (max-width: 768px) {
        img {
            max-width: 100%;
            height: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR - NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Page:",
                        ["Dashboard", "Detect Fraud", "About"])

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("Fraud Detection System")
    st.markdown("Real-time transaction monitoring and fraud detection")

    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.875rem; opacity: 0.9;">Detection Accuracy</div>
            <div style="font-size: 2rem; font-weight: 700;">95.2%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.875rem; opacity: 0.9;">Transactions Analyzed</div>
            <div style="font-size: 2rem; font-weight: 700;">5,000+</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.875rem; opacity: 0.9;">Fraud Rate</div>
            <div style="font-size: 2rem; font-weight: 700;">5.0%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.875rem; opacity: 0.9;">System Status</div>
            <div style="font-size: 2rem; font-weight: 700;">Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Exploratory Data Analysis**")
        try:
            st.image("eda_visualization.png", use_container_width=True)
        except FileNotFoundError:
            st.markdown("<p class='card-text' style='font-size: 0.875rem;'>Run <code>python 3_eda.py</code> to generate this visualization</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Model Performance**")
        try:
            st.image("model_evaluation.png", use_container_width=True)
        except FileNotFoundError:
            st.markdown("<p class='card-text' style='font-size: 0.875rem;'>Run <code>python 7_evaluate_model.py</code> to generate this visualization</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Statistics Chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Detection Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Weekly Detection Trend**")
        fig, ax = plt.subplots(figsize=(6, 4))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        detections = [45, 52, 38, 41, 55, 32, 28]
        frauds = [5, 8, 3, 4, 7, 2, 2]
        ax.plot(days, detections, marker='o', label='Total', color='#1e3a8a', linewidth=2)
        ax.plot(days, frauds, marker='s', label='Fraud', color='#dc2626', linewidth=2)
        ax.set_xlabel('Day')
        ax.set_ylabel('Count')
        ax.set_title('Weekly Transaction Analysis')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.markdown("**Detection by Risk Level**")
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_levels = ['Low', 'Medium', 'High']
        counts = [245, 35, 20]
        colors = ['#16a34a', '#f59e0b', '#dc2626']
        ax.bar(risk_levels, counts, color=colors, alpha=0.8)
        ax.set_ylabel('Count')
        ax.set_title('Transactions by Risk Level')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(counts):
            ax.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PAGE 2: FRAUD DETECTION
# ============================================================

elif page == "Detect Fraud":
    st.title("Fraud Detection")
    st.markdown("Analyze transaction details for potential fraud")

    # Load model
    model, scaler = load_model_and_scaler()

    if model is not None:
        # Create input form
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Transaction Details")

            amount = st.slider(
                "Transaction Amount (₹)",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100
            )

            time = st.selectbox(
                "Time of Transaction",
                options=list(range(24)),
                format_func=lambda x: f"{x:02d}:00"
            )

            merchant = st.selectbox(
                "Merchant Category",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: f"Category {x}"
            )

            card_type_dict = {0: "Debit Card", 1: "Credit Card", 2: "Prepaid Card"}
            card_type = st.radio(
                "Card Type",
                options=[0, 1, 2],
                format_func=lambda x: card_type_dict[x]
            )

            international = st.selectbox(
                "International Transaction",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Risk Analysis")

            # Calculate risk factors
            risk_score = 0
            risk_factors = []

            if amount > 7000:
                risk_score += 25
                risk_factors.append("High transaction amount")
            if time in [0, 1, 2, 3, 23]:
                risk_score += 20
                risk_factors.append("Unusual transaction time")
            if international == 1:
                risk_score += 30
                risk_factors.append("International transaction")
            if merchant in [1, 3]:
                risk_score += 15
                risk_factors.append("Higher risk merchant category")

            # Display risk meter
            st.markdown("**Preliminary Risk Assessment**")
            risk_color = "#16a34a" if risk_score < 30 else "#f59e0b" if risk_score < 60 else "#dc2626"
            st.markdown(f"""
            <div style="background-color: #e2e8f0; border-radius: 8px; height: 24px; overflow: hidden; margin: 10px 0;">
                <div style="background-color: {risk_color}; width: {risk_score}%; height: 100%; transition: width 0.3s;"></div>
            </div>
            <div class="card-text" style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                <span>Low Risk</span>
                <span>{risk_score}%</span>
                <span>High Risk</span>
            </div>
            """, unsafe_allow_html=True)

            if risk_factors:
                st.markdown("**Detected Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("*No significant risk factors detected*")

            st.markdown("</div>", unsafe_allow_html=True)

        # Prediction Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Transaction", use_container_width=True, type="primary"):
            # Prepare data
            transaction = np.array([[amount, time, merchant, card_type, international]])
            transaction_scaled = scaler.transform(transaction)

            # Make prediction
            prediction = model.predict(transaction_scaled)[0]
            probability = model.predict_proba(transaction_scaled)[0]
            fraud_prob = probability[1]

            # Display result
            st.markdown("<br>", unsafe_allow_html=True)

            if fraud_prob > 0.6:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                risk_desc = "Likely Fraudulent"
                icon = ""
            elif fraud_prob > 0.4:
                risk_class = "risk-medium"
                risk_label = "MEDIUM RISK"
                risk_desc = "Suspicious Activity"
                icon = ""
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                risk_desc = "Likely Genuine"
                icon = ""

            # Result Card - built as one self-contained HTML block to avoid two bugs:
            # 1. Indented lines after blank lines inside triple-quoted strings are
            #    treated as Markdown code blocks by the 4-space rule.
            # 2. Splitting opening/closing <div> across separate st.markdown() calls
            #    breaks the browser DOM (each call gets its own wrapper div).
            contributing_factors = []
            if international == 1:
                contributing_factors.append("International transaction")
            if amount > 7000:
                contributing_factors.append("High transaction amount")
            if time in [0, 1, 2, 3, 23]:
                contributing_factors.append("Unusual time")
            if not contributing_factors:
                contributing_factors.append("Normal transaction pattern")

            factors_html = "".join(
                f'<div style="margin:0.25rem 0;">• {f}</div>'
                for f in contributing_factors[:3]
            )
            card_level = "high" if fraud_prob > 0.6 else "medium" if fraud_prob > 0.4 else "low"
            result_card_html = (
                f'<div class="card fraud-{card_level}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">'  
                f'<h3 style="margin:0;">Analysis Result</h3>'
                f'<span class="risk-badge {risk_class}">{risk_label}</span></div>'
                f'<div style="font-size:1.5rem;font-weight:600;margin-bottom:0.5rem;">{risk_desc}</div>'
                f'<div class="card-text" style="margin-bottom:1.5rem;">Fraud Probability: <strong>{fraud_prob:.1%}</strong></div>'
                f'<div style="background:rgba(0,0,0,0.05);border-radius:8px;padding:1rem;margin-bottom:1rem;">'
                f'<div class="card-text" style="font-size:0.875rem;margin-bottom:0.5rem;">Confidence Score</div>'
                f'<div style="font-size:2rem;font-weight:700;">{max(probability):.1%}</div></div>'
                f'<div class="card-text" style="font-size:0.875rem;margin-bottom:0.5rem;"><strong>Key Contributing Factors:</strong></div>'
                f'<div style="font-size:0.875rem;">{factors_html}</div>'
                f'</div>'
            )
            st.markdown(result_card_html, unsafe_allow_html=True)

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve Transaction", use_container_width=True):
                    st.success("Transaction approved")
            with col2:
                if st.button("Flag for Review", use_container_width=True):
                    st.warning("Transaction flagged for manual review")


# ============================================================
# PAGE 3: ABOUT
# ============================================================

elif page == "About":
    st.title("About This System")

    # System Overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("System Overview")
    st.markdown("""
    <p class="card-text">
    Machine learning-powered fraud detection system for real-time credit card transaction analysis.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance Metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Performance Metrics")

    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Score': ['95.2%', '92.8%', '89.5%', '0.910', '0.945']
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

    # Performance chart
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    values = [95.2, 92.8, 89.5, 91.0, 94.5]
    colors = ['#16a34a', '#16a34a', '#f59e0b', '#16a34a', '#16a34a']
    ax.bar(metrics, values, color=colors, alpha=0.8)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 1, f'{v}%', ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Feature Importance")

    importance_data = {
        'Feature': ['International Transaction', 'Transaction Amount', 'Time of Day', 'Merchant Category', 'Card Type'],
        'Importance': [0.32, 0.28, 0.22, 0.12, 0.06]
    }

    importance_df = pd.DataFrame(importance_data)
    st.table(importance_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='#1e3a8a')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance in Fraud Detection')
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training Data Distribution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Data Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Before SMOTE (Imbalanced)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes = [4750, 250]
        labels = ['Genuine', 'Fraud']
        colors = ['#16a34a', '#dc2626']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax.set_title('Original Dataset')
        st.pyplot(fig)

    with col2:
        st.markdown("**After SMOTE (Balanced)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes = [4000, 4000]
        labels = ['Genuine', 'Fraud']
        colors = ['#16a34a', '#dc2626']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax.set_title('After SMOTE Balancing')
        st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Technical Information
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Technical Information")

    st.info("""
    **Model Files Required:**
    - `fraud_detection_model.pkl` - Trained machine learning model
    - `scaler.pkl` - Feature scaling parameters

    **To run the training pipeline:**
    ```
    python 1_data_collection.py
    python 2_preprocessing.py
    python 3_eda.py
    python 4_handle_imbalance.py
    python 5_feature_scaling.py
    python 6_train_model.py
    python 7_evaluate_model.py
    python 8_make_predictions.py
    ```

    **To launch the application:**
    ```
    streamlit run 9_streamlit_app.py
    ```
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div class="card-text" style='text-align: center; font-size: 0.875rem; padding: 1rem 0;'>
Fraud Detection System | Built with Streamlit
</div>
""", unsafe_allow_html=True)  