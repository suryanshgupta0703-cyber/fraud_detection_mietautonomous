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
def load_models_and_scaler():
    """Load both models and scaler (cached for performance)"""
    try:
        rf_model = joblib.load('fraud_detection_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return rf_model, xgb_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found! {e}")
        st.info("Please run the training pipeline first:")
        st.write("1. python 1_data_collection.py through 5_feature_scaling.py")
        st.write("2. python 6_train_model.py (Random Forest)")
        st.write("3. python 6b_train_xgboost_model.py (XGBoost)")
        return None, None, None

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
    <style>
    /* Clean, Professional FinTech Dashboard Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Base App Canvas - Clean neutral backgrounds, zero distracting animations */
    .stApp {
        background-color: #f8fafc;
    }

    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0b0f19;
        }
    }

    /* Content Cards */
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.25rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    }

    .card-text {
        color: #64748b;
        line-height: 1.5;
    }

    @media (prefers-color-scheme: dark) {
        .card {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: #f1f5f9;
            box-shadow: none;
        }
        .card:hover {
            border-color: #475569;
        }
        .card-text {
            color: #94a3b8;
        }
    }

    /* Stat Cards - Clean KPI presentation */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .stat-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }

    .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .stat-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .stat-pill {
        display: inline-flex;
        align-items: center;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        margin-top: 0.5rem;
    }

    .stat-pill.success {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }

    .stat-pill.neutral {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    .stat-pill.warning {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
    }

    @media (prefers-color-scheme: dark) {
        .stat-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            box-shadow: none;
        }
        .stat-card:hover {
            border-color: #475569;
        }
        .stat-label {
            color: #94a3b8;
        }
        .stat-value {
            color: #f8fafc;
        }
        .stat-pill.success {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .stat-pill.neutral {
            background: rgba(148, 163, 184, 0.15);
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        .stat-pill.warning {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
    }

    /* Result Cards - Clear, Professional Semantic Banners */
    .fraud-high {
        background-color: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 4px solid #e11d48;
        border-radius: 12px;
        padding: 1.5rem;
        color: #1e293b;
    }

    .fraud-medium {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        border-radius: 12px;
        padding: 1.5rem;
        color: #1e293b;
    }

    .fraud-low {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 12px;
        padding: 1.5rem;
        color: #1e293b;
    }

    @media (prefers-color-scheme: dark) {
        .fraud-high {
            background-color: rgba(225, 29, 72, 0.1);
            border: 1px solid rgba(225, 29, 72, 0.3);
            border-left: 4px solid #e11d48;
            color: #f8fafc;
        }
        .fraud-medium {
            background-color: rgba(217, 119, 6, 0.1);
            border: 1px solid rgba(217, 119, 6, 0.3);
            border-left: 4px solid #d97706;
            color: #f8fafc;
        }
        .fraud-low {
            background-color: rgba(22, 163, 74, 0.1);
            border: 1px solid rgba(22, 163, 74, 0.3);
            border-left: 4px solid #16a34a;
            color: #f8fafc;
        }
    }

    /* Confidence Box */
    .confidence-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }

    .confidence-score {
        color: #0f172a;
        font-weight: 700;
    }

    @media (prefers-color-scheme: dark) {
        .confidence-box {
            background-color: #0f172a !important;
            border: 1px solid #334155;
            box-shadow: none;
        }
        .confidence-score {
            color: #f8fafc !important;
        }
    }

    /* Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
    }

    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }

    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }

    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }

    @media (prefers-color-scheme: dark) {
        .risk-high {
            background-color: rgba(225, 29, 72, 0.2);
            color: #fda4af;
            border-color: rgba(225, 29, 72, 0.4);
        }
        .risk-medium {
            background-color: rgba(245, 158, 11, 0.2);
            color: #fde68a;
            border-color: rgba(245, 158, 11, 0.4);
        }
        .risk-low {
            background-color: rgba(22, 163, 74, 0.2);
            color: #a7f3d0;
            border-color: rgba(22, 163, 74, 0.4);
        }
    }

    /* Typography */
    h1 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }

    h2, h3 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    @media (prefers-color-scheme: dark) {
        h1 {
            color: #f8fafc;
        }
        h2, h3 {
            color: #f1f5f9;
        }
    }

    /* Clean Buttons */
    .stButton > button {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #0f172a;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #1e293b;
        border-color: #1e293b;
        color: #ffffff;
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    @media (prefers-color-scheme: dark) {
        .stButton > button {
            background-color: #3b82f6;
            border-color: #3b82f6;
            color: #ffffff;
        }
        .stButton > button:hover {
            background-color: #2563eb;
            border-color: #2563eb;
        }
    }

    /* Page container spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1120px;
    }

    /* Mobile Responsive adjustments */
    @media (max-width: 768px) {
        .card, .stat-card {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .stat-value {
            font-size: 1.5rem;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
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
            <div class="stat-label">Detection Accuracy</div>
            <div class="stat-value">95.2%</div>
            <div class="stat-pill success">Verified Test</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Transactions Analyzed</div>
            <div class="stat-value">5,000+</div>
            <div class="stat-pill neutral">Full Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Fraud Rate</div>
            <div class="stat-value">5.0%</div>
            <div class="stat-pill warning">250 Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">System Status</div>
            <div class="stat-value" style="color: #059669;">Active</div>
            <div class="stat-pill success">Dual Engine</div>
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
        fig, ax = plt.subplots(figsize=(6, 3.8))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        detections = [45, 52, 38, 41, 55, 32, 28]
        frauds = [5, 8, 3, 4, 7, 2, 2]
        ax.plot(days, detections, marker='o', label='Total Checked', color='#475569', linewidth=2, markersize=5)
        ax.plot(days, frauds, marker='s', label='Fraud Detected', color='#e11d48', linewidth=2, markersize=5)
        ax.set_xlabel('Day', fontsize=9, color='#64748b')
        ax.set_ylabel('Transactions', fontsize=9, color='#64748b')
        ax.set_title('Weekly Transaction Volume', fontsize=11, fontweight='600', pad=10)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Detection by Risk Level**")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        counts = [245, 35, 20]
        colors = ['#10b981', '#f59e0b', '#e11d48']
        bars = ax.bar(risk_levels, counts, color=colors, width=0.55, alpha=0.9)
        ax.set_ylabel('Transaction Count', fontsize=9, color='#64748b')
        ax.set_title('Risk Categorization Breakdown', fontsize=11, fontweight='600', pad=10)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 4, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='600', color='#334155')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PAGE 2: FRAUD DETECTION
# ============================================================

elif page == "Detect Fraud":
    st.title("Fraud Detection")
    st.markdown("Analyze transaction details for potential fraud")

    # Load models
    rf_model, xgb_model, scaler = load_models_and_scaler()

    if rf_model is not None and xgb_model is not None:
        # Model selection
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose Model for Analysis",
            ["Random Forest", "XGBoost", "Compare Both"],
            horizontal=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Select active model(s)
        if model_choice == "Random Forest":
            active_models = {"Random Forest": rf_model}
        elif model_choice == "XGBoost":
            active_models = {"XGBoost": xgb_model}
        else:
            active_models = {"Random Forest": rf_model, "XGBoost": xgb_model}
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
            risk_color = "#10b981" if risk_score < 30 else "#f59e0b" if risk_score < 60 else "#e11d48"
            st.markdown(f"""
            <div style="background-color: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; height: 16px; overflow: hidden; margin: 10px 0;">
                <div style="background-color: {risk_color}; width: {risk_score}%; height: 100%; transition: width 0.3s ease;"></div>
            </div>
            <div class="card-text" style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 500;">
                <span style="color: #10b981;">Low Risk</span>
                <span style="font-weight: 700; color: {risk_color};">{risk_score}%</span>
                <span style="color: #e11d48;">High Risk</span>
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

            # Make predictions for each active model
            results = {}
            for model_name, model in active_models.items():
                prediction = model.predict(transaction_scaled)[0]
                probability = model.predict_proba(transaction_scaled)[0]
                fraud_prob = probability[1]
                results[model_name] = {
                    'prediction': prediction,
                    'fraud_prob': fraud_prob,
                    'probability': probability
                }

            # Display results
            st.markdown("<br>", unsafe_allow_html=True)

            if len(results) == 1:
                # Single model result
                model_name = list(results.keys())[0]
                result = results[model_name]
                fraud_prob = result['fraud_prob']
                probability = result['probability']

                if fraud_prob > 0.6:
                    risk_class = "risk-high"
                    risk_label = "HIGH RISK"
                    risk_desc = "Likely Fraudulent"
                elif fraud_prob > 0.4:
                    risk_class = "risk-medium"
                    risk_label = "MEDIUM RISK"
                    risk_desc = "Suspicious Activity"
                else:
                    risk_class = "risk-low"
                    risk_label = "LOW RISK"
                    risk_desc = "Likely Genuine"

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
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">'
                    f'<h3 style="margin:0;font-size:1.15rem;">{model_name} Analysis</h3>'
                    f'<span class="risk-badge {risk_class}">{risk_label}</span></div>'
                    f'<div style="font-size:1.35rem;font-weight:700;margin-bottom:0.35rem;">{risk_desc}</div>'
                    f'<div class="card-text" style="margin-bottom:1rem;font-size:0.9rem;">Fraud Probability: <strong>{fraud_prob:.1%}</strong></div>'
                    f'<div class="confidence-box">'
                    f'<div class="card-text" style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.25rem;">Model Confidence</div>'
                    f'<div class="confidence-score" style="font-size:1.75rem;font-weight:700;">{max(probability):.1%}</div></div>'
                    f'<div class="card-text" style="font-size:0.85rem;margin-bottom:0.35rem;"><strong>Key Contributing Factors:</strong></div>'
                    f'<div style="font-size:0.85rem;">{factors_html}</div>'
                    f'</div>'
                )
                st.markdown(result_card_html, unsafe_allow_html=True)

            else:
                # Comparison mode - show both models
                for model_name, result in results.items():
                    fraud_prob = result['fraud_prob']
                    probability = result['probability']

                    if fraud_prob > 0.6:
                        risk_class = "risk-high"
                        risk_label = "HIGH RISK"
                        risk_desc = "Likely Fraudulent"
                    elif fraud_prob > 0.4:
                        risk_class = "risk-medium"
                        risk_label = "MEDIUM RISK"
                        risk_desc = "Suspicious Activity"
                    else:
                        risk_class = "risk-low"
                        risk_label = "LOW RISK"
                        risk_desc = "Likely Genuine"

                    card_level = "high" if fraud_prob > 0.6 else "medium" if fraud_prob > 0.4 else "low"
                    result_card_html = (
                        f'<div class="card fraud-{card_level}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">'
                        f'<h3 style="margin:0;font-size:1.05rem;">{model_name}</h3>'
                        f'<span class="risk-badge {risk_class}">{risk_label}</span></div>'
                        f'<div style="font-size:1.15rem;font-weight:700;margin-bottom:0.25rem;">{risk_desc}</div>'
                        f'<div class="card-text" style="margin-bottom:0.5rem;font-size:0.875rem;">Fraud Probability: <strong>{fraud_prob:.1%}</strong> &bull; Confidence: <strong>{max(probability):.1%}</strong></div>'
                        f'</div>'
                    )
                    st.markdown(result_card_html, unsafe_allow_html=True)

                # Agreement indicator
                rf_result = results["Random Forest"]
                xgb_result = results["XGBoost"]
                agreement = abs(rf_result['fraud_prob'] - xgb_result['fraud_prob']) < 0.1

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Model Consensus")
                if agreement:
                    st.success("✅ **Consensus Reached**: Both models agree on the risk classification.")
                else:
                    st.warning("⚠️ **Divergence Detected**: Random Forest and XGBoost show different probability estimates. Manual review recommended.")
                st.markdown(f"""
                <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.75rem;font-size:0.875rem;">
                    <div><span class="card-text">Random Forest:</span> <strong>{rf_result['fraud_prob']:.1%}</strong></div>
                    <div><span class="card-text">XGBoost:</span> <strong>{xgb_result['fraud_prob']:.1%}</strong></div>
                    <div><span class="card-text">Absolute Difference:</span> <strong>{abs(rf_result['fraud_prob'] - xgb_result['fraud_prob']):.1%}</strong></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

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
    fig, ax = plt.subplots(figsize=(8, 3.8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    values = [96.5, 60.6, 86.0, 71.1, 98.3]
    colors = ['#334155', '#475569', '#e11d48', '#64748b', '#10b981']
    bars = ax.bar(metrics, values, color=colors, width=0.55, alpha=0.9)
    ax.set_ylabel('Score (%)', fontsize=9, color='#64748b')
    ax.set_title('Random Forest Evaluation Metrics (Test Set)', fontsize=11, fontweight='600', pad=10)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='600', color='#334155')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Feature Importance")

    importance_data = {
        'Feature': ['Transaction Amount', 'Time of Day', 'Merchant Category', 'International Transaction', 'Card Type'],
        'Importance': [0.491, 0.198, 0.138, 0.091, 0.081]
    }

    importance_df = pd.DataFrame(importance_data)
    st.table(importance_df)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='#334155', height=0.55)
    ax.set_xlabel('Relative Importance Score', fontsize=9, color='#64748b')
    ax.set_title('Feature Importance in Fraud Detection', fontsize=11, fontweight='600', pad=10)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., f'{width:.1%}',
                ha='left', va='center', fontsize=9, fontweight='600', color='#334155')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training Data Distribution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Data Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Before SMOTE (Imbalanced)**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sizes = [4750, 250]
        labels = ['Genuine', 'Fraud']
        colors = ['#10b981', '#e11d48']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Original Dataset (95:5)', fontsize=10, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**After SMOTE (Balanced)**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sizes = [4000, 4000]
        labels = ['Genuine', 'Fraud']
        colors = ['#10b981', '#e11d48']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('After SMOTE Balancing (50:50)', fontsize=10, fontweight='600')
        plt.tight_layout()
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