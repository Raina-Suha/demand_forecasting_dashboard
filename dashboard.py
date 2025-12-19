# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("forecast_with_alerts_and_suggestions.csv", parse_dates=['date'])

st.title("🛒 Retail Sales Forecast & Inventory Alerts")

st.subheader("📈 Sales Forecast vs Actual")
fig = px.line(df, x='date', y=['sales', 'predicted_sales'], labels={'value': 'Sales'}, title="Actual vs Predicted Sales")
st.plotly_chart(fig, use_container_width=True)

st.subheader("⚠️ Inventory Alerts")
st.markdown("### 🔔 Understocking Alerts")
st.dataframe(df[df['understock_alert']][['date', 'stock', 'predicted_sales']])

st.markdown("### 📦 Overstocking Alerts")
st.dataframe(df[df['overstock_alert']][['date', 'stock', 'predicted_sales']])

st.subheader("💡 Suggestions to Improve Sales")
st.dataframe(df[df['suggestions'].notnull()][['date', 'predicted_sales', 'explanations', 'suggestions']])
