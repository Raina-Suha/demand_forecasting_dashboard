#  Demand Forecasting Dashboard

An end-to-end **Demand Forecasting Dashboard** that predicts future demand from historical data and generates **alerts and actionable suggestions** to support better inventory and business decisions.

---

##  Features
- Time-series demand forecasting
- Automated alerts for demand spikes and drops
- Actionable inventory suggestions
- SHAP-based model explainability
- Visual dashboard for insights
- Scheduled forecasting automation

---

##  Forecasting Methodology
The model analyzes historical demand patterns to forecast future demand values.  
It identifies trends and unusual changes in demand to flag potential business risks.

To improve transparency, **SHAP (SHapley Additive exPlanations)** is used to explain how different features influence predictions.

---

##  Alerts & Suggestions
The dashboard automatically detects:
- Sudden increases in demand
- Sharp demand drops
- Possible overstock or stockout scenarios

Based on these signals, the system provides suggestions such as:
- Increasing inventory
- Monitoring demand closely
- Adjusting supply strategies

---

##  How to Run

### Install dependencies
```bash
pip install -r requirements.txt
