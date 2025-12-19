import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('sample_data.csv', parse_dates=['Date'])

# Rename and clean column names for easier handling
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Check for required columns
required_columns = ['Inventory_Level', 'Units_Sold', 'Date', 'Holiday_Promotion']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Sort by date (important for time series)
df.sort_values('Date', inplace=True)

# Feature engineering
df['dayofweek'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month

# Convert 'Holiday_Promotion' into binary flags
df['promo'] = df['Holiday_Promotion'].astype(str).str.contains('Promotion', case=False, na=False).astype(int)
df['holiday'] = df['Holiday_Promotion'].astype(str).str.contains('Holiday', case=False, na=False).astype(int)

# Features & target
features = ['promo', 'holiday', 'dayofweek', 'month']
target = 'Units_Sold'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
df['predicted_sales'] = model.predict(X)

# Evaluate
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # Fixed line
print(f"\n📉 Test RMSE: {rmse:.2f}")

# SHAP explainability
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Save SHAP summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary.png")

# Inventory alerts
df['understock_alert'] = df['Inventory_Level'] < df['predicted_sales']
df['overstock_alert'] = df['Inventory_Level'] > df['predicted_sales'] * 1.5  # Adjustable threshold

# Filter low sales predictions (bottom 25%)
low_sales_threshold = df['predicted_sales'].quantile(0.25)
low_sales_df = df[df['predicted_sales'] <= low_sales_threshold]

# Get SHAP values using positional indexing to avoid indexing error
low_shap_indices = df.index.get_indexer(low_sales_df.index)
low_shap_values = shap_values[low_shap_indices]

# Function to generate explanations and suggestions
def generate_suggestions(shap_row, feature_row):
    explanations = []
    suggestions = []

    for i, feature in enumerate(features):
        shap_val = shap_row[i]
        value = feature_row[feature]

        if shap_val < 0:  # Negative impact
            explanations.append(f"{feature} reduced predicted sales.")
            if feature == 'promo' and value == 0:
                suggestions.append("Run a promotion to boost sales.")
            elif feature == 'holiday' and value == 1:
                suggestions.append("Holiday may have lowered demand — adjust strategy.")
            elif feature == 'dayofweek' and value in [0, 1]:
                suggestions.append("Sales are low early in the week — consider weekday offers.")
            elif feature == 'month' and value in [1, 2]:
                suggestions.append("Consider campaigns for off-season months like Jan/Feb.")
        elif shap_val > 0:
            explanations.append(f"{feature} increased predicted sales.")

    return explanations, suggestions

# Apply to low-sales rows
explanations_list = []
suggestions_list = []

for i, idx in enumerate(low_sales_df.index):
    shap_row = low_shap_values[i].values
    feature_row = df.loc[idx, features]
    explanations, suggestions = generate_suggestions(shap_row, feature_row)
    explanations_list.append("; ".join(explanations))
    suggestions_list.append("; ".join(set(suggestions)))  # Avoid duplicate suggestions

# Add explanations/suggestions
df.loc[low_sales_df.index, 'explanations'] = explanations_list
df.loc[low_sales_df.index, 'suggestions'] = suggestions_list

# Display alerts
print("\n🔔 Understocking Alerts:")
print(df[df['understock_alert']][['Date', 'Inventory_Level', 'predicted_sales']])

print("\n📦 Overstocking Alerts:")
print(df[df['overstock_alert']][['Date', 'Inventory_Level', 'predicted_sales']])

# Display low sales recommendations
print("\n📉 Low Sales Days — Explanations and Suggestions:")
print(df[df['predicted_sales'] <= low_sales_threshold][['Date', 'predicted_sales', 'explanations', 'suggestions']])

# Save output
df.to_csv("forecast_with_alerts_and_suggestions.csv", index=False)
print("\n✅ Forecast, alerts, and suggestions saved to 'forecast_with_alerts_and_suggestions.csv'")
