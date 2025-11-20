import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic the dark/sleek look
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

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # 1. Try the local path provided by the user
    local_path = "C:/Users/TIRELESS/Desktop/Housing Loans Board Site/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    # 2. Fallback URL (Standard Telco Churn Dataset)
    public_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    df = pd.DataFrame()
    
    try:
        df = pd.read_csv(local_path)
    except FileNotFoundError:
        try:
            df = pd.read_csv(public_url)
        except Exception:
            return pd.DataFrame() 

    if df.empty:
        return df

    # --- PREPROCESSING ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
    return df

df = load_data()

if df.empty:
    st.error("Could not load data. Please check your local file path or internet connection.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS (FILTERS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛠️ Dashboard Controls")
    st.write("Filter the data to explore specific segments.")
    st.markdown("---")
    
    # -- Categorical Filters --
    st.subheader("Demographics & Services")
    
    contract_options = ['All'] + sorted(df['Contract'].unique().tolist())
    selected_contract = st.selectbox("Contract Type", contract_options)
    
    internet_options = ['All'] + sorted(df['InternetService'].unique().tolist())
    selected_internet = st.selectbox("Internet Service", internet_options)
    
    payment_options = ['All'] + sorted(df['PaymentMethod'].unique().tolist())
    selected_payment = st.selectbox("Payment Method", payment_options)
    
    tech_options = ['All'] + sorted(df['TechSupport'].unique().tolist())
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


# -----------------------------------------------------------------------------
# 4. FILTERING LOGIC
# -----------------------------------------------------------------------------
df_filtered = df.copy()

# Apply Categorical Filters
if selected_contract != 'All':
    df_filtered = df_filtered[df_filtered['Contract'] == selected_contract]
if selected_internet != 'All':
    df_filtered = df_filtered[df_filtered['InternetService'] == selected_internet]
if selected_payment != 'All':
    df_filtered = df_filtered[df_filtered['PaymentMethod'] == selected_payment]
if selected_tech != 'All':
    df_filtered = df_filtered[df_filtered['TechSupport'] == selected_tech]

# Apply Numerical Filters
df_filtered = df_filtered[
    (df_filtered['tenure'] >= selected_tenure[0]) & 
    (df_filtered['tenure'] <= selected_tenure[1])
]
df_filtered = df_filtered[
    (df_filtered['MonthlyCharges'] >= selected_charges[0]) & 
    (df_filtered['MonthlyCharges'] <= selected_charges[1])
]


# -----------------------------------------------------------------------------
# 5. MAIN APP NAVIGATION (TABS)
# -----------------------------------------------------------------------------
st.title("📊 Customer Churn Prediction Dashboard")

# Create the main navigation tabs
tab_overview, tab_performance, tab_student = st.tabs(["Overview", "Model Performance", "Student Info"])

# === TAB 1: OVERVIEW ===
with tab_overview:
    # --- TOP KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df_filtered)
    num_churned = len(df_filtered[df_filtered['Churn'] == 'Yes'])
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
        if not df_filtered.empty:
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
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available for selected filters.")
        
    with c2:
        st.subheader("Churn Rate by Contract Type")
        if not df_filtered.empty:
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
            # Sample data if it's too large for a quick scatter plot
            plot_df = df_filtered.sample(1000) if len(df_filtered) > 1000 else df_filtered
            
            fig_scatter = px.scatter(
                plot_df,
                x="tenure",
                y="MonthlyCharges",
                color="Churn",
                color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'},
                opacity=0.6,
                hover_data=['Contract', 'PaymentMethod']
            )
            fig_scatter.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Monthly Charges ($)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    with c4:
        st.write("**Churn Distribution by Tenure**")
        if not df_filtered.empty:
            fig_hist = px.histogram(
                df_filtered, 
                x="tenure", 
                color="Churn", 
                nbins=30,
                color_discrete_map={'No': '#1F77B4', 'Yes': '#D62728'},
                opacity=0.7,
                labels={"tenure": "Tenure (Months)"}
            )
            fig_hist.update_layout(barmode='overlay')
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- ROW 3: Payment Method Analysis ---
    st.subheader("Payment Method Impact")
    if not df_filtered.empty:
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
        st.write("**Feature Importance**")
        features = ['Contract', 'Tenure', 'MonthlyCharges', 'TechSupport', 'OnlineSecurity']
        importance = [0.35, 0.25, 0.15, 0.15, 0.10]
        
        df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
        fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
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
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150) 
        with c2:
            st.markdown("### Student Profile")
            st.write("**Name:** [Student Name]")
            st.write("**Student ID:** [12345678]")
            st.write("**Department:** Data Science & Analytics")
            st.write("**Institution:** [University Name]")
            
    st.success("This project demonstrates the application of data visualization and dashboarding techniques to solve real-world business problems.")
