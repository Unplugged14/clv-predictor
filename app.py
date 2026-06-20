import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('clv_model.pkl')

RECOMMENDATIONS = {
    'Champion':  "🏆 VIP treatment — early access, loyalty rewards, referral program",
    'Loyal':     "💛 Upsell premium products, offer subscription plans",
    'At Risk':   "⚡ Send win-back campaign with personalised discount within 7 days",
    'Lost':      "💤 Low priority — small reactivation email, move budget elsewhere"
}

st.set_page_config(page_title="CLV Predictor", page_icon="💰", layout="wide")
st.title("💰 Customer Lifetime Value Predictor")
st.caption("Enter a customer\'s behaviour metrics to predict their 90-day value and segment.")

tab1, tab2 = st.tabs(["🔍 Single customer", "📁 Batch prediction (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        recency  = st.slider("Recency (days since last purchase)", 1, 365, 30)
        freq     = st.slider("Frequency (number of orders)", 1, 100, 5)
        monetary = st.slider("Monetary (total spend £)", 10, 10000, 500)
    with col2:
        avg_order  = st.number_input("Avg order value (£)", value=round(monetary/freq, 2))
        purch_span = st.slider("Purchase span (days)", 0, 365, 90)

    if st.button("Predict CLV", type="primary"):
        log_monetary = np.log1p(monetary)
        X = pd.DataFrame([[recency, freq, log_monetary, avg_order, purch_span]],
                          columns=['Recency', 'Frequency', 'Log_Monetary',
                                   'AvgOrderValue', 'PurchaseSpan'])
        log_pred = model.predict(X)[0]
        clv_pred = np.expm1(log_pred)

        if clv_pred > 500:   segment = "Champion"
        elif clv_pred > 200: segment = "Loyal"
        elif clv_pred > 50:  segment = "At Risk"
        else:                segment = "Lost"

        st.metric("Predicted 90-day CLV", f"£{clv_pred:.2f}")
        st.info(f"**Segment:** {segment}")
        st.success(RECOMMENDATIONS[segment])

with tab2:
    uploaded = st.file_uploader("Upload customer CSV", type="csv")
    st.caption("CSV must contain columns: Recency, Frequency, Monetary, AvgOrderValue, PurchaseSpan")
    if uploaded:
        batch = pd.read_csv(uploaded)
        batch['Log_Monetary'] = np.log1p(batch['Monetary'])
        X_batch = batch[['Recency', 'Frequency', 'Log_Monetary',
                          'AvgOrderValue', 'PurchaseSpan']]
        batch['Predicted_CLV_GBP'] = np.expm1(model.predict(X_batch))
        st.dataframe(batch)
        st.download_button("Download predictions", batch.to_csv(index=False),
                           "clv_predictions.csv")
