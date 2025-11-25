# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic the dark/sleek look (kept as you had it)
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    /* Style for the Tabs to make them look more like the screenshot */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.1);
        border-bottom: 2px solid #FF4B4B;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    # Use the dataset path you specified
    data_path = "/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # If the file is not at that exact path, try current dir fallback
    if not os.path.exists(data_path):
        fallback = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if os.path.exists(fallback):
            data_path = fallback

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        # Try the public fallback
        public_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        try:
            df = pd.read_csv(public_url)
        except Exception:
            st.error("Could not load data. Please ensure /WA_Fn-UseC_-Telco-Customer-Churn.csv exists or you have internet access.")
            return pd.DataFrame()

    # Basic preprocessing used for display and subsequent transforms
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

    # SeniorCitizen may be 0/1 integer; also map to 'Yes'/'No' for display consistency if needed
    if 'SeniorCitizen' in df.columns:
        # Keep numeric column as-is but also create readable label for display consistency
        try:
            # If column contains 0/1 integers
            if set(df['SeniorCitizen'].dropna().unique()).issubset({0, 1}):
                df['SeniorCitizen_label'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        except Exception:
            pass

    return df

df = load_data()
if df.empty:
    st.stop()

# ---------------------------------------------------------------------
# 3. MODEL & FEATURE LOADING
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    # load model
    model_path = "lgbm_churn_model.joblib"
    features_path = "model_features.joblib"

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in app directory.")
        st.stop()
    if not os.path.exists(features_path):
        st.error(f"Feature list '{features_path}' not found in app directory.")
        st.stop()

    model = joblib.load(model_path)
    model_features = joblib.load(features_path)
    return model, model_features

model, model_features = load_model_and_features()

# ---------------------------------------------------------------------
# 4. PREDICTION PREPROCESSING FUNCTION (must match training transforms)
# ---------------------------------------------------------------------
def preprocess_input(df_input, model_features):
    """
    Applies preprocessing + feature engineering in the SAME ORDER used during training.
    Ensures result columns exactly match model_features (in the same order).
    """
    df = df_input.copy()

    # Ensure typical columns exist (fill with defaults if missing)
    expected_cols = [
        "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines",
        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0 if c in ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"] else "No"

    # TotalCharges numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # SeniorCitizen: accept 'Yes'/'No' or 1/0
    def norm_sen(x):
        if pd.isna(x): return 0
        if isinstance(x, str):
            return 1 if x.strip().lower() in ["yes","1","true","y"] else 0
        try:
            return int(x)
        except:
            return 0
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(norm_sen)

    # has_internet_service
    df["has_internet_service"] = df["InternetService"].apply(lambda x: 1 if str(x) in ["DSL", "Fiber optic"] else 0)

    # monthly_to_total_charges_ratio
    df["monthly_to_total_charges_ratio"] = (df["TotalCharges"] / df["MonthlyCharges"]).replace([np.inf, -np.inf], 0).fillna(0)

    # has_multiple_services (count Yes across service columns)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    # normalize service responses
    for c in service_cols:
        df[c] = df[c].astype(str)
    df["has_multiple_services"] = df[service_cols].apply(lambda row: int(sum([1 if v.strip().lower()=='yes' else 0 for v in row]) > 2), axis=1)

    # is_senior_partner_dependent
    df["Partner"] = df["Partner"].astype(str)
    df["Dependents"] = df["Dependents"].astype(str)
    df["is_senior_partner_dependent"] = ((df["SeniorCitizen"] == 1) & (df["Partner"].str.lower() == "yes") & (df["Dependents"].str.lower() == "yes")).astype(int)

    # One-hot encode all categorical columns (drop_first=True to mirror training)
    # Determine categorical columns to encode (exclude numerical engineered features)
    categorical_cols = [
        "gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Add any missing model features as zeros, and ensure ordering
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only model_features columns (and in same order)
    df_final = df[model_features].copy()

    return df_final

# ---------------------------------------------------------------------
# 5. SIDEBAR CONTROLS (FILTERS) - keep your UI logic unchanged
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🛠️ Dashboard Controls")
    st.write("Filter the data to explore specific segments.")
    st.markdown("---")

    # -- Categorical Filters --
    st.subheader("Demographics & Services")
    contract_options = ['All'] + sorted(df['Contract'].astype(str).unique().tolist())
    selected_contract = st.selectbox("Contract Type", contract_options)

    internet_options = ['All'] + sorted(df['InternetService'].astype(str).unique().tolist())
    selected_internet = st.selectbox("Internet Service", internet_options)

    payment_options = ['All'] + sorted(df['PaymentMethod'].astype(str).unique().tolist())
    selected_payment = st.selectbox("Payment Method", payment_options)

    # Some datasets have 'TechSupport' missing values; fill for filter options
    tech_col = df['TechSupport'].astype(str).unique().tolist()
    tech_options = ['All'] + sorted(tech_col)
    selected_tech = st.selectbox("Tech Support", tech_options)

    st.markdown("---")

    # -- Numerical Filters --
    st.subheader("Quantitative Filters")

    # Tenure Slider
    min_tenure = int(df['tenure'].min())
    max_tenure = int(df['tenure'].max())
    selected_tenure = st.slider("Tenure (Months)", min_tenure, max_tenure, (min_tenure, max_tenure))

    # Monthly Charges Slider
    min_charge = float(df['MonthlyCharges'].min())
    max_charge = float(df['MonthlyCharges'].max())
    selected_charges = st.slider("Monthly Charges ($)", min_charge, max_charge, (min_charge, max_charge))

# ---------------------------------------------------------------------
# 6. FILTERING LOGIC (same as provided)
# ---------------------------------------------------------------------
df_filtered = df.copy()

if selected_contract != 'All':
    df_filtered = df_filtered[df_filtered['Contract'] == selected_contract]
if selected_internet != 'All':
    df_filtered = df_filtered[df_filtered['InternetService'] == selected_internet]
if selected_payment != 'All':
    df_filtered = df_filtered[df_filtered['PaymentMethod'] == selected_payment]
if selected_tech != 'All':
    df_filtered = df_filtered[df_filtered['TechSupport'] == selected_tech]

df_filtered = df_filtered[
    (df_filtered['tenure'] >= selected_tenure[0]) &
    (df_filtered['tenure'] <= selected_tenure[1])
]
df_filtered = df_filtered[
    (df_filtered['MonthlyCharges'] >= selected_charges[0]) &
    (df_filtered['MonthlyCharges'] <= selected_charges[1])
]

# ---------------------------------------------------------------------
# 7. MAIN APP NAVIGATION (TABS) - added Predict tab at the end
# ---------------------------------------------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
tab_overview, tab_performance, tab_student, tab_predict = st.tabs(["Overview", "Model Performance", "Student Info", "🔮 Predict Churn"])

# === TAB 1: OVERVIEW ===
with tab_overview:
    # --- TOP KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df_filtered)
    num_churned = len(df_filtered[df_filtered['Churn'] == 'Yes']) if 'Churn' in df_filtered.columns else 0
    churn_rate = (num_churned / total_customers * 100) if total_customers > 0 else 0
    avg_monthly = df_filtered['MonthlyCharges'].mean() if total_customers > 0 else 0

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Customers Churned", f"{num_churned:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

    st.markdown("---")

    # --- ROW 1: Churn Distribution & Contract Type ---
    c1, c2 = st.columns((1, 2))

    with c1:
        st.subheader("Churn Distribution")
        if not df_filtered.empty and 'Churn' in df_filtered.columns:
            churn_counts = df_filtered['Churn'].value_counts().reset_index()
            churn_counts.columns = ['Churn', 'Count']

            fig_pie = px.pie(
                churn_counts,
                names='Churn',
                values='Count',
                color='Churn',
                color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'},
                hole=0.5
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, width='stretch')
        else:
            st.info("No data available for selected filters.")

    with c2:
        st.subheader("Churn Rate by Contract Type")
        if not df_filtered.empty and 'Churn' in df_filtered.columns:
            contract_churn = df_filtered.groupby(['Contract', 'Churn']).size().reset_index(name='Count')

            fig_bar = px.bar(
                contract_churn,
                x='Contract',
                y='Count',
                color='Churn',
                barmode='group',
                color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'},
                text_auto=True
            )
            fig_bar.update_layout(xaxis_title="Contract Type", yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data available.")

    # --- ROW 2: Scatter Plot (New Insight) & Tenure ---
    st.subheader("Advanced Insights")
    c3, c4 = st.columns(2)

    with c3:
        st.write("**Risk Analysis: Monthly Charges vs. Tenure**")
        st.caption("High charges with low tenure often indicate higher churn risk.")
        if not df_filtered.empty:
            plot_df = df_filtered.sample(1000) if len(df_filtered) > 1000 else df_filtered

            fig_scatter = px.scatter(
                plot_df,
                x="tenure",
                y="MonthlyCharges",
                color="Churn" if 'Churn' in plot_df.columns else None,
                color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'},
                opacity=0.6,
                hover_data=['Contract', 'PaymentMethod'] if 'Contract' in plot_df.columns else None
            )
            fig_scatter.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Monthly Charges ($)")
            st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        st.write("**Churn Distribution by Tenure**")
        if not df_filtered.empty and 'tenure' in df_filtered.columns:
            fig_hist = px.histogram(
                df_filtered,
                x="tenure",
                color="Churn" if 'Churn' in df_filtered.columns else None,
                nbins=30,
                color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'},
                opacity=0.7,
                labels={"tenure": "Tenure (Months)"}
            )
            fig_hist.update_layout(barmode='overlay')
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- ROW 3: Payment Method Analysis ---
    st.subheader("Payment Method Impact")
    if not df_filtered.empty and 'PaymentMethod' in df_filtered.columns and 'Churn' in df_filtered.columns:
        payment_churn = df_filtered.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
        fig_pay = px.bar(
            payment_churn,
            x='PaymentMethod',
            y='Count',
            color='Churn',
            barmode='stack',
            color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'}
        )
        st.plotly_chart(fig_pay, use_container_width=True)

# === TAB 2: MODEL PERFORMANCE ===
with tab_performance:
    st.subheader("🤖 Model Prediction Performance")
    st.markdown("Performance metrics of the Machine Learning model used to predict customer churn.")
    st.info("Note: This section uses generic data to demonstrate how model metrics would be displayed.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Confusion Matrix**")
        z = [[1200, 300], [200, 800]]
        x = ['Predicted No', 'Predicted Yes']
        y = ['Actual No', 'Actual Yes']

        fig_cm = px.imshow(z, x=x, y=y, text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.write("**Feature Importance (example)**")
        # If your model exposes feature importance, you can replace these example values.
        try:
            fi = model.feature_importances_
            fi_features = model_features[:len(fi)]
            df_imp = pd.DataFrame({"Feature": fi_features, "Importance": fi})
            df_imp = df_imp.sort_values("Importance", ascending=True)
            fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            features = ['Contract', 'Tenure', 'MonthlyCharges', 'TechSupport', 'OnlineSecurity']
            importance = [0.35, 0.25, 0.15, 0.15, 0.10]
            df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
            fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Model Accuracy Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "85.2%")
    m2.metric("Precision", "82.1%")
    m3.metric("Recall", "79.5%")
    m4.metric("F1 Score", "80.8%")

# === TAB 3: STUDENT INFO ===
with tab_student:
    st.subheader("🎓 Student Information")

    with st.container():
        st.markdown("### Project Details")
        st.markdown("""
        **Research Topic:** *Predictive Analytics in Telecommunications: Mitigating Customer Churn using Machine Learning*
        """)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            # show placeholder image if file missing
            if os.path.exists("ekene.png"):
                st.image("ekene.png", width=150)
            else:
                st.write("Image 'ekene.png' not found.")
        with c2:
            st.markdown("### Student Profile")
            st.write("**Name:** Ekene Olise")
            st.write("**Student ID:** 2303926")
            st.write("**Department:** Information Technology with Business Intelligence")
            st.write("School of Computing Engineering and Technology (SOCET)")
            st.write("**Institution:** Robert Gordon University (RGU) Aberdeen Scotland UK")

    st.success("This project demonstrates the application of data visualization and dashboarding techniques to solve real-world business problems.")

# === TAB 4: PREDICT CHURN (NEW) ===
with tab_predict:
    st.subheader("🔮 Real-Time Churn Prediction")
    st.write("Enter customer information to predict churn probability.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            depend = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (Months)", 0, 72, 1)

        with col2:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

            monthly = st.number_input("Monthly Charges", 0.0, 2000.0, 70.0, step=0.1)
            total = st.number_input("Total Charges", 0.0, 100000.0, 0.0, step=0.1)

        submit = st.form_submit_button("Predict")

    if submit:
        # Build input row
        input_dict = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": depend,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": "No",            # default (not in form)
            "DeviceProtection": "No",        # default
            "TechSupport": tech,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        input_df = pd.DataFrame([input_dict])

        # Preprocess to match model features
        processed = preprocess_input(input_df, model_features)

        # Predict probability
        try:
            prob = model.predict_proba(processed)[0][1]
            pred_label = "Yes" if prob > 0.5 else "No"

            st.success(f"**Predicted Churn:** {pred_label}")
            st.info(f"**Churn Probability:** {prob:.2%}")

            # Optionally show top contributing features if SHAP available (not required)
            # If you want SHAP explanations added, I can attach code to compute and display SHAP values.
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
