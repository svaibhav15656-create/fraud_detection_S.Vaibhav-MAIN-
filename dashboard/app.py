
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, shap, json, warnings, matplotlib.pyplot as plt, matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artefacts ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load("dashboard/model.pkl")
    scaler    = joblib.load("dashboard/scaler.pkl")
    threshold = joblib.load("dashboard/threshold.pkl")
    with open("feature_cols.json") as f:
        feat_cols = json.load(f)
    return model, scaler, threshold, feat_cols

@st.cache_data
def load_data():
    return pd.read_csv("test_results.csv")

model, scaler, threshold, feat_cols = load_model()
df = load_data()

TIER_COLOR = {
    " Critical Risk": "#e74c3c",
    " Suspicious"   : "#f39c12",
    " Clear"        : "#27ae60",
}

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
st.sidebar.title(" Fraud Detection")
page = st.sidebar.radio("Navigate", [" Overview", " Transaction Explorer", " SHAP Explainer"])
st.sidebar.markdown("---")

tier_filter = st.sidebar.multiselect(
    "Filter by Risk Tier",
    options=df["RiskTier"].unique().tolist(),
    default=df["RiskTier"].unique().tolist()
)
amt_min, amt_max = float(df["TransactionAmt"].min()), float(df["TransactionAmt"].max())
amt_range = st.sidebar.slider("Transaction Amount Range ($)",
                               amt_min, amt_max, (amt_min, amt_max))

df_filtered = df[
    (df["RiskTier"].isin(tier_filter)) &
    (df["TransactionAmt"].between(*amt_range))
].copy()


# PAGE 1 — OVERVIEW

if page == " Overview":
    st.title(" Fraud Detection — Operations Dashboard")
    st.caption("Real-time fraud monitoring powered by LightGBM + SHAP")
    st.markdown("---")

    # KPI metrics
    total        = len(df_filtered)
    total_fraud  = df_filtered["isFraud"].sum()
    detect_rate  = df_filtered[df_filtered["isFraud"]==1]["fraud_prob"].apply(
                     lambda p: p >= threshold).mean() * 100
    avg_fraud_amt = df_filtered[df_filtered["isFraud"]==1]["TransactionAmt"].mean()
    critical_cnt = (df_filtered["RiskTier"] == " Critical Risk").sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Confirmed Fraud",    f"{int(total_fraud):,}")
    c3.metric("Detection Rate",     f"{detect_rate:.1f}%")
    c4.metric("Avg Fraud Amount",   f"${avg_fraud_amt:,.2f}")
    c5.metric(" Critical Alerts",  f"{int(critical_cnt):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_cnt = df_filtered["RiskTier"].value_counts().reset_index()
        tier_cnt.columns = ["Tier","Count"]
        fig_donut = px.pie(tier_cnt, names="Tier", values="Count",
                           color="Tier",
                           color_discrete_map=TIER_COLOR,
                           hole=0.5,
                           title="Transaction Risk Tiers")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.subheader("Fraud Rate by Hour of Day")
        hour_g = (df_filtered.groupby("HourOfDay")["isFraud"]
                  .mean().reset_index().rename(columns={"isFraud":"FraudRate"}))
        fig_hour = px.bar(hour_g, x="HourOfDay", y="FraudRate",
                          title="Fraud Rate by Hour",
                          labels={"FraudRate":"Fraud Rate","HourOfDay":"Hour"},
                          color="FraudRate", color_continuous_scale="Reds")
        st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("---")
    st.subheader("Transaction Amount — Fraud vs Legitimate (Scatter)")
    sample = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
    fig_sc = px.scatter(
        sample, x="HourOfDay", y="TransactionAmt",
        color="fraud_prob",
        color_continuous_scale="RdYlGn_r",
        size="TransactionAmt",
        size_max=20,
        hover_data=["RiskTier","isFraud","fraud_prob"],
        title="TransactionAmt vs HourOfDay (coloured by Fraud Probability)",
        labels={"fraud_prob":"Fraud Prob","HourOfDay":"Hour","TransactionAmt":"Amount ($)"}
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# PAGE 2 — TRANSACTION EXPLORER
elif page == " Transaction Explorer":
    st.title(" Transaction Explorer")
    st.caption("Search and filter individual transactions with live risk scores")
    st.markdown("---")

    search_id = st.text_input(" Search by TransactionID", placeholder="e.g. 2987000")

    if search_id:
        try:
            sid = int(search_id)
            row = df_filtered[df_filtered.get("TransactionID", pd.Series()) == sid]
            if len(row):
                st.success(f"Found TransactionID {sid}")
                st.dataframe(row)
            else:
                st.warning("TransactionID not found in filtered dataset.")
        except:
            st.error("Enter a valid numeric TransactionID.")

    st.subheader(" Transaction Table")
    show_cols = ["TransactionAmt","HourOfDay","DayOfWeek",
                 "fraud_prob","RiskTier","isFraud"]
    show_cols = [c for c in show_cols if c in df_filtered.columns]

    st.dataframe(
        df_filtered[show_cols].sort_values("fraud_prob", ascending=False).head(500),
        use_container_width=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud Probability Distribution")
        fig_hist = px.histogram(df_filtered, x="fraud_prob", nbins=60,
                                color="RiskTier",
                                color_discrete_map=TIER_COLOR,
                                title="Fraud Probability Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        st.subheader("Avg Amount by Risk Tier")
        amt_tier = (df_filtered.groupby("RiskTier")["TransactionAmt"]
                    .mean().reset_index())
        fig_bar = px.bar(amt_tier, x="RiskTier", y="TransactionAmt",
                         color="RiskTier", color_discrete_map=TIER_COLOR,
                         title="Avg Transaction Amount by Risk Tier",
                         labels={"TransactionAmt":"Avg Amount ($)"})
        st.plotly_chart(fig_bar, use_container_width=True)


# PAGE 3 — SHAP EXPLAINER

elif page == " SHAP Explainer":
    st.title(" SHAP Explainer — Individual Transaction")
    st.caption("Enter a row index to see why the model made its prediction")
    st.markdown("---")

    row_idx = st.number_input("Enter row index (0 to N-1)",
                               min_value=0,
                               max_value=len(df_filtered)-1,
                               value=0, step=1)

    if st.button(" Explain This Transaction"):
        row_data = df_filtered.iloc[row_idx]
        prob     = row_data.get("fraud_prob", 0.0)
        tier     = row_data.get("RiskTier", "")
        actual   = int(row_data.get("isFraud", -1))

        cols = st.columns(3)
        cols[0].metric("Fraud Probability", f"{prob:.2%}")
        cols[1].metric("Risk Tier",          tier)
        cols[2].metric("Actual Label",       "FRAUD" if actual==1 else "LEGITIMATE")

        # Get raw features for this row
        raw_feat_cols = [c for c in feat_cols if c in df_filtered.columns]
        if not raw_feat_cols:
            st.error("Feature columns not available in the results table.")
        else:
            row_features = df_filtered.iloc[[row_idx]][raw_feat_cols]
            row_scaled   = pd.DataFrame(scaler.transform(row_features),
                                        columns=raw_feat_cols)

            with st.spinner("Computing SHAP values ..."):
                explainer = shap.TreeExplainer(model)
                sv_row    = explainer(row_scaled)

                if hasattr(sv_row, "values") and sv_row.values.ndim == 3:
                    sv_plot = sv_row[0, :, 1]
                else:
                    sv_plot = sv_row[0]

            st.subheader("SHAP Waterfall Plot")
            fig_shap, ax = plt.subplots(figsize=(12, 6))
            shap.plots.waterfall(sv_plot, max_display=15, show=False)
            plt.title(f"SHAP Explanation — Row {row_idx}", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

            # Plain-English
            shap_ser = pd.Series(sv_plot.values, index=sv_plot.feature_names)
            top5 = shap_ser.abs().nlargest(5).index.tolist()
            st.subheader(" Plain-English Explanation")
            if prob >= 0.75:
                verdict = " HIGH RISK — Recommend immediate review / block."
            elif prob >= 0.40:
                verdict = " SUSPICIOUS — Flag for manual analyst review."
            else:
                verdict = "LOW RISK — Transaction appears legitimate."

            st.info(f"""
**Verdict:** {verdict}

**Fraud Probability:** {prob:.2%}

**Top 5 contributing features:**
{chr(10).join([f"- `{f}`: SHAP = {shap_ser[f]:+.4f}" for f in top5])}

**Interpretation:**
Features with positive SHAP values *increase* fraud probability.
Features with negative SHAP values *decrease* fraud probability.
The model base rate for fraud is approximately {df["isFraud"].mean()*100:.2f}%.
""")
