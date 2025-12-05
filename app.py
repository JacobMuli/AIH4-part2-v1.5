import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Option A — NDVI Yield Intelligence", layout="wide")
st.title("🌱 Option A — NDVI-Based Potato Yield Forecasting")

# ------------------------------
# FILE UPLOAD
# ------------------------------
st.sidebar.header("Upload Cleaned Data & Model")
data_file = st.sidebar.file_uploader("Upload ndvi_optionb_cleaned.csv", type=["csv"])
model_file = st.sidebar.file_uploader("Upload rf_optionb_ndvi_model.joblib", type=["joblib"])

if not data_file or not model_file:
    st.info("Please upload cleaned dataset and trained model to continue.")
    st.stop()

df = pd.read_csv(data_file)
rf = joblib.load(model_file)

df.columns = df.columns.str.lower().str.strip()

# ------------------------------
# COUNTY SELECTION
# ------------------------------
counties = sorted(df["admin_1"].dropna().unique())
county = st.sidebar.selectbox("Select County", counties)

# Filter
cdf = df[df["admin_1"] == county].copy().sort_values("harvest_year")

st.header(f"County Selected: **{county}**")
st.write("Dataset (filtered):")
st.dataframe(cdf.head())

# ------------------------------
# BASIC EDA
# ------------------------------
st.subheader("📊 Yield Over Time")

if "yield" in cdf.columns:
    by_year = cdf.groupby("harvest_year")["yield"].mean()
    fig, ax = plt.subplots()
    ax.plot(by_year.index, by_year.values, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (t/ha)")
    ax.set_title("Mean Yield per Year")
    st.pyplot(fig)
else:
    st.warning("Column 'yield' missing from dataset.")

# ------------------------------
# AGGREGATION FOR FORECASTING
# ------------------------------
agg = cdf.groupby("harvest_year").agg({
    "area": "sum",
    "production": "sum",
    "mean_annual_ndvi": "mean"
}).reset_index()

agg["yield"] = agg["production"] / agg["area"]

# ------------------------------
# FEATURE RECONSTRUCTION FOR MODEL
# ------------------------------
def build_feature_vector(last_row, rf_model):
    """
    Recreate the exact feature vector the model expects.
    Handles:
    - mean_annual_ndvi
    - area
    - planting_month default=1
    - county dummy variables (if model used them)
    """
    base = {
        "mean_annual_ndvi": float(last_row.get("mean_annual_ndvi", 0.0)),
        "area": float(last_row.get("area", 0.0)),
        "planting_month": 1  # assume Jan planting
    }

    # If model has feature_names_in_, rebuild exactly
    if hasattr(rf_model, "feature_names_in_"):
        expected = list(rf_model.feature_names_in_)
        row = {}

        for feat in expected:
            if feat in base:
                row[feat] = base[feat]
            elif feat.startswith("adm1_"):  # county dummy
                row[feat] = 1.0 if feat == f"adm1_{last_row.get('admin_1','')}" else 0.0
            else:
                row[feat] = 0.0  # unused features set to zero

        X = np.array([[row[f] for f in expected]], dtype=float)
        return X

    # Fallback: model was trained only on numeric base features
    return np.array([[base["mean_annual_ndvi"], base["area"], base["planting_month"]]], dtype=float)

# ------------------------------
# MULTI-YEAR FORECASTING
# ------------------------------
st.header("📅 Multi-Year Forecasting")

last_year = int(agg["harvest_year"].max())
current_year = datetime.datetime.now().year

target_year = st.number_input(
    "Forecast up to year:",
    min_value=last_year + 1,
    max_value=current_year,
    value=last_year + 1,
)

def recursive_forecast(history_df, rf_model, final_year):
    hist = history_df.copy().reset_index(drop=True)
    results = []

    for year in range(int(hist["harvest_year"].max()) + 1, final_year + 1):
        last = hist.iloc[-1]

        # Build correct feature vector
        X = build_feature_vector(last, rf_model)

        try:
            pred_yield = float(rf_model.predict(X)[0])
        except Exception as e:
            raise ValueError(f"Forecast failed: {e}")

        area = float(last["area"])
        pred_prod = pred_yield * area

        results.append({
            "year": year,
            "predicted_yield": pred_yield,
            "predicted_production": pred_prod
        })

        # Expand history for next iteration
        new_row = {
            "harvest_year": year,
            "area": area,
            "production": pred_prod,
            "mean_annual_ndvi": last["mean_annual_ndvi"],  # assume stable NDVI
            "yield": pred_yield,
            "admin_1": last.get("admin_1", None)
        }
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(results)

forecast_df = recursive_forecast(agg, rf, int(target_year))

st.subheader("Forecast Results")
st.dataframe(forecast_df)

final_tonnes = float(forecast_df.iloc[-1]["predicted_production"])

# ------------------------------
# STORAGE PLANNING ENGINE
# ------------------------------
st.header("❄ Cold Storage Requirement")

def pack_storage(tonnes, sizes=[1000, 500, 250], fill=0.9):
    needed = int(np.ceil(tonnes / fill))
    allocation = []
    remaining = needed

    for s in sizes:
        c = remaining // s
        if c > 0:
            allocation.append({"size": s, "count": c})
            remaining -= c * s

    if remaining > 0:
        allocation.append({"size": 250, "count": 1})

    total = sum(x["size"] * x["count"] for x in allocation)
    util = tonnes / total

    return {"required_capacity": needed, "allocation": allocation, "utilization": util}

storage = pack_storage(final_tonnes)
st.json(storage)

# ------------------------------
# DOWNLOAD OUTPUT
# ------------------------------
st.download_button(
    "Download Forecast Output",
    data=str({"forecast": forecast_df.to_dict(), "storage": storage}),
    file_name="optionA_forecast_output.json"
)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import datetime

# st.set_page_config(page_title="Option B — NDVI Yield Intelligence", layout="wide")
# st.title("🌱 Option B — NDVI-Based Potato Yield Forecasting (Multi-County)")

# st.sidebar.header("Upload Cleaned Data & Model")
# cleaned_file = st.sidebar.file_uploader("Upload ndvi_optionb_cleaned.csv", type=["csv"])
# model_file = st.sidebar.file_uploader("Upload rf_optionb_ndvi_model.joblib", type=["joblib"])

# if cleaned_file is None or model_file is None:
#     st.info("Please upload both cleaned CSV and model to continue.")
#     st.stop()

# df = pd.read_csv(cleaned_file)
# rf = joblib.load(model_file)

# # County selector
# counties = sorted(df["admin_1"].dropna().unique())
# county = st.sidebar.selectbox("County", counties)

# # Filter dataset
# cdf = df[df["admin_1"] == county].copy()
# cdf = cdf.sort_values("harvest_year")

# st.header(f"County Selected: {county}")
# st.dataframe(cdf.head())

# # EDA Plot
# st.subheader("Yield over time")
# grp = cdf.groupby("harvest_year")["yield"].mean()
# fig, ax = plt.subplots()
# ax.plot(grp.index, grp.values, marker="o")
# ax.set_xlabel("Year")
# ax.set_ylabel("Yield (t/ha)")
# st.pyplot(fig)

# # Prepare aggregated dataset
# agg = cdf.groupby("harvest_year").agg({
#     "area":"sum",
#     "production":"sum",
#     "mean_annual_ndvi":"mean"
# }).reset_index()
# agg["yield"] = agg["production"] / agg["area"]

# # Multi-year forecasting
# st.header("📅 Multi-year Forecasting")

# last_year = int(agg["harvest_year"].max())
# current_year = datetime.datetime.now().year

# target_year = st.number_input(
#     "Forecast up to year",
#     min_value=last_year + 1,
#     max_value=current_year,
#     value=last_year + 1,
#     step=1
# )

# def recursive_forecast(df, rf_model, final_year):
#     df_hist = df.copy()
#     results = []

#     for year in range(int(df_hist["harvest_year"].max()) + 1, final_year + 1):
#         mean_ndvi = df_hist["mean_annual_ndvi"].iloc[-1]
#         area = df_hist["area"].iloc[-1]

#         X = np.array([[mean_ndvi, area, 1]])  # planting_month=1
#         pred_yield = rf_model.predict(X)[0]
#         pred_production = pred_yield * area

#         results.append({
#             "year": year,
#             "predicted_yield": pred_yield,
#             "predicted_production": pred_production
#         })

#         # Append to history
#         df_hist.loc[len(df_hist)] = [year, area, pred_production, mean_ndvi, pred_yield]

#     return pd.DataFrame(results)

# forecast_df = recursive_forecast(agg, rf, int(target_year))
# st.subheader("Forecasted Results")
# st.dataframe(forecast_df)

# final_row = forecast_df.iloc[-1]

# # Storage Engine
# st.header("❄ Storage Requirement")

# def pack_storage(tonnes, sizes=[1000,500,250], fill=0.9):
#     required = int(np.ceil(tonnes / fill))
#     allocation = []
#     remaining = required

#     for size in sizes:
#         count = remaining // size
#         if count > 0:
#             allocation.append({"size":size, "count":count})
#             remaining -= size * count

#     if remaining > 0:
#         allocation.append({"size":250, "count":1})

#     total = sum(a["size"]*a["count"] for a in allocation)
#     util = tonnes / total

#     return {"required_capacity":required,"allocation":allocation,"utilization":util}

# storage = pack_storage(final_row["predicted_production"])
# st.json(storage)

# st.download_button(
#     "Download Results",
#     data=str({"forecast":forecast_df.to_dict(),"storage":storage}),
#     file_name="optionb_forecast_output.json"
# )
