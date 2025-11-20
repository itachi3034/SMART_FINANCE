import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# Make sure script runs from its own folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 1. Load the Data ---
try:
    df = pd.read_csv('bulk_investment_data.csv')
except FileNotFoundError:
    print("Error: 'bulk_investment_data.csv' not found. Please ensure it's in the correct folder.")
    raise SystemExit

# If your CSV still uses USD column names, rename them to INR for internal consistency
rename_map = {}
if 'Initial_Cost_USD' in df.columns:
    rename_map['Initial_Cost_USD'] = 'Initial_Cost_INR'
if 'Scrap_Value_USD' in df.columns:
    rename_map['Scrap_Value_USD'] = 'Scrap_Value_INR'
if 'ANNUAL_OPERATING_COST_USD' in df.columns:
    rename_map['ANNUAL_OPERATING_COST_USD'] = 'ANNUAL_OPERATING_COST_INR'

if rename_map:
    df = df.rename(columns=rename_map)

# --- 2. Data Preprocessing: Convert Machine_Model (Text) to Numbers ---
df = pd.get_dummies(df, columns=['Machine_Model'], drop_first=True)

# --- 3. Define Features (X) and Multiple Targets (y) ---
target_cols = [
    'Actual_Lifespan_Yrs',
    'Historical_Total_Maint_Cost',
    'Task_Completion_Time_Hrs'
]
# Basic safety check
for t in target_cols:
    if t not in df.columns:
        raise KeyError(f"Expected target column '{t}' not found in CSV.")

X = df.drop(columns=target_cols)
y = df[target_cols]

# --- 4. Split the Data ---
# (original used test_size=0.8; keeping that for compatibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 5. Initialize and Train the Multi-Output Model ---
base_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
multi_output_model = MultiOutputRegressor(base_estimator)

print("\nStarting model training...")
multi_output_model.fit(X_train, y_train)
print("Model training complete!")

# --- 6. Evaluate the Model (Per Target) ---
y_pred_test = multi_output_model.predict(X_test)

print("\n--- Model Performance on Test Data ---")
for i, target in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_test[:, i])
    print(f"Target: {target}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2 Score): {r2:.4f}")

# ============================================================
#  USER-DRIVEN PREDICTION + FINANCIAL COMPARISON SECTION (INR)
# ============================================================

# Helper: safely get numeric input
def get_float(prompt_text):
    while True:
        try:
            return float(input(prompt_text))
        except ValueError:
            print("Please enter a valid number.")

def get_int(prompt_text):
    while True:
        try:
            return int(input(prompt_text))
        except ValueError:
            print("Please enter a valid integer.")

# --- 7. Collect User Inputs for Machine Options ---
print("\n=======================================================")
print("        USER INPUT: MACHINE OPTIONS TO COMPARE (₹)     ")
print("=======================================================")

n_machines = get_int("How many machine options do you want to compare? ")

user_rows = []
for i in range(n_machines):
    print(f"\n--- Machine {i+1} ---")
    machine_name = input("Enter machine model/name (e.g., 'Lathe X-A1'): ").strip()

    initial_cost = get_float("Initial cost (₹): ")
    scrap_value = get_float("Expected scrap/salvage value at end of life (₹): ")
    gst_rate_pct = get_float("GST tax rate (%) on purchase (e.g., 18 for 18%): ")

    power = get_float("Power consumption (kWh per hour): ")
    capacity = get_float("Max capacity (units per hour): ")
    maint_interval = get_float("Scheduled maintenance interval (days): ")
    failure_rate = get_float("Historical failure rate (0–1): ")

    user_rows.append({
        "Machine_Model": machine_name,
        "Initial_Cost_INR": initial_cost,
        "Scrap_Value_INR": scrap_value,
        "GST_Rate_Pct": gst_rate_pct,
        "Power_Consumption_kWh_hr": power,
        "Max_Capacity_Units_hr": capacity,
        "Scheduled_Maint_Interval_Days": maint_interval,
        "Historical_Failure_Rate": failure_rate
    })

new_machine_data = pd.DataFrame(user_rows)

# --- 8. Use the ML model to predict outputs for user machines ---

# Columns used by the model (must match training features)
# NOTE: these names must align with X.columns; we use the INR names for costs
model_input_cols = [
    "Initial_Cost_INR",
    "Power_Consumption_kWh_hr",
    "Max_Capacity_Units_hr",
    "Scheduled_Maint_Interval_Days",
    "Historical_Failure_Rate",
    "Machine_Model"
]

# Make a copy of just the columns the model expects
model_input_df = new_machine_data[model_input_cols].copy()

# Apply same encoding as training
model_input_df = pd.get_dummies(model_input_df, columns=['Machine_Model'], drop_first=True)

# Align columns with training data (X) — fill missing columns with 0
model_input_df = model_input_df.reindex(columns=X.columns, fill_value=0)

# Predict
predicted_outputs = multi_output_model.predict(model_input_df)
predictions_df = pd.DataFrame(predicted_outputs, columns=target_cols)

# Merge predictions back with user data
comparison_df = new_machine_data.copy()
comparison_df['Predicted_Lifespan'] = predictions_df['Actual_Lifespan_Yrs']
comparison_df['Predicted_Maint_Cost'] = predictions_df['Historical_Total_Maint_Cost']
comparison_df['Predicted_Task_Time'] = predictions_df['Task_Completion_Time_Hrs']

print("\n--- ML Predicted Outputs for Your Inputs (INR system) ---")
print(predictions_df.assign(Machine_Model=new_machine_data["Machine_Model"]).to_string(index=False))

# --- 9. Get Financial Parameters from User ---
print("\n=======================================================")
print("      USER INPUT: FINANCIAL PARAMETERS (GLOBAL, ₹)     ")
print("=======================================================")

BULK_QUANTITY = get_int("Bulk quantity of units to purchase: ")
TIME_HORIZON_YRS = get_int("Time horizon for comparison (years): ")
TARGET_REVENUE_PER_UNIT_YR = get_float("Expected annual revenue per unit (₹): ")
ANNUAL_OPERATING_COST_INR = get_float("Annual operating cost per unit (₹): ")

# --- 10. Financial Metrics Calculation ---

# Effective initial cost including GST and scrap:
# Effective Initial = Initial_Cost * (1 + GST%) - Scrap_Value
comparison_df['GST_Factor'] = 1 + (comparison_df['GST_Rate_Pct'] / 100.0)
comparison_df['Effective_Initial_Cost'] = (
    comparison_df['Initial_Cost_INR'] * comparison_df['GST_Factor']
    - comparison_df['Scrap_Value_INR']
)

# Avoid division by zero when computing Annual_Maint_Cost
# If predicted lifespan <= 0, set annual maint to NaN (or handle as you prefer)
comparison_df['Predicted_Lifespan_clean'] = comparison_df['Predicted_Lifespan'].replace({0: np.nan})
comparison_df['Annual_Maint_Cost'] = (
    comparison_df['Predicted_Maint_Cost'] / comparison_df['Predicted_Lifespan_clean']
)

# If Predicted_Lifespan was NaN, fill Annual_Maint_Cost with predicted value divided by TIME_HORIZON_YRS (fallback)
comparison_df['Annual_Maint_Cost'] = comparison_df['Annual_Maint_Cost'].fillna(
    comparison_df['Predicted_Maint_Cost'] / np.where(TIME_HORIZON_YRS > 0, TIME_HORIZON_YRS, 1)
)

# Total Cost of Ownership (TCO) over time horizon
comparison_df['TCO_Per_Unit'] = (
    comparison_df['Effective_Initial_Cost']
    + (ANNUAL_OPERATING_COST_INR * TIME_HORIZON_YRS)
    + (comparison_df['Annual_Maint_Cost'] * TIME_HORIZON_YRS)
)

# Total revenue & net profit
comparison_df['Total_Revenue_Per_Unit'] = TARGET_REVENUE_PER_UNIT_YR * TIME_HORIZON_YRS
comparison_df['Net_Profit_Per_Unit'] = (
    comparison_df['Total_Revenue_Per_Unit'] - comparison_df['TCO_Per_Unit']
)

# ROI relative to effective initial cost. If Effective_Initial_Cost <= 0 use NaN
comparison_df['ROI_Yrs'] = np.where(
    comparison_df['Effective_Initial_Cost'] > 0,
    (comparison_df['Net_Profit_Per_Unit'] / comparison_df['Effective_Initial_Cost']) * 100.0,
    np.nan
)

# Scale to bulk quantity
comparison_df['Bulk_Total_TCO'] = comparison_df['TCO_Per_Unit'] * BULK_QUANTITY
comparison_df['Bulk_Total_Profit'] = comparison_df['Net_Profit_Per_Unit'] * BULK_QUANTITY

# --- 11. Final Recommendation ---

# If all Net_Profit_Per_Unit are NaN or equal, handle gracefully
if comparison_df['Net_Profit_Per_Unit'].isnull().all():
    print("Warning: Net profit could not be computed for any machine. Check predictions and inputs.")
    best_model_row = comparison_df.iloc[0]
else:
    best_model_row = comparison_df.loc[comparison_df['Net_Profit_Per_Unit'].idxmax()]

print("\n=======================================================")
print("  💰 FINANCIAL COMPARISON (INR system) 💰")
print("=======================================================")
print(
    f"ASSUMPTIONS: Bulk Quantity={BULK_QUANTITY}, "
    f"Time Horizon={TIME_HORIZON_YRS} years, "
    f"Annual Revenue/Unit=₹{TARGET_REVENUE_PER_UNIT_YR:,.2f}, "
    f"Annual Operating Cost/Unit=₹{ANNUAL_OPERATING_COST_INR:,.2f}"
)

# Build a nice summary table
summary_cols = [
    'Machine_Model',
    'Initial_Cost_INR',
    'Scrap_Value_INR',
    'GST_Rate_Pct',
    'Predicted_Lifespan',
    'Annual_Maint_Cost',
    'Net_Profit_Per_Unit',
    'ROI_Yrs'
]

summary = comparison_df[summary_cols].copy()
summary.rename(columns={
    'Machine_Model': 'Machine',
    'Initial_Cost_INR': 'Initial Cost (₹)',
    'Scrap_Value_INR': 'Scrap Value (₹)',
    'GST_Rate_Pct': 'GST (%)',
    'Predicted_Lifespan': 'Pred. Life (Yrs)',
    'Annual_Maint_Cost': 'Annual Maint. (₹)',
    'Net_Profit_Per_Unit': 'Net Profit/Unit (₹)',
    'ROI_Yrs': 'ROI (%)'
}, inplace=True)

# Format currency columns with rupee symbol
summary['Initial Cost (₹)'] = summary['Initial Cost (₹)'].map('₹{:,.0f}'.format)
summary['Scrap Value (₹)'] = summary['Scrap Value (₹)'].map('₹{:,.0f}'.format)
summary['Annual Maint. (₹)'] = summary['Annual Maint. (₹)'].map('₹{:,.0f}'.format)
summary['Net Profit/Unit (₹)'] = summary['Net Profit/Unit (₹)'].map('₹{:,.0f}'.format)

# Format ROI - handle NaN
summary['ROI (%)'] = summary['ROI (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

print("\n--- Comparative Financial Metrics (Per Unit) ---")
print(summary.to_string(index=False))

print("\n--- FINAL INVESTMENT RECOMMENDATION ---")
print(f"Recommended machine for a bulk investment of {BULK_QUANTITY} units: {best_model_row['Machine_Model']}")
print(
    f"Projected Net Profit over {TIME_HORIZON_YRS} years (bulk): "
    f"₹{best_model_row['Bulk_Total_Profit']:,.0f}"
)
# Print ROI using computed column
roi_display = best_model_row['ROI_Yrs']
roi_text = f"{roi_display:.2f}%" if pd.notnull(roi_display) else "N/A"
print(f"Estimated ROI over {TIME_HORIZON_YRS} years: {roi_text}")
print("=======================================================")

