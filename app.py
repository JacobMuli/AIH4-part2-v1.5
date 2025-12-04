import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Option B — NDVI Yield Intelligence", layout="wide")
st.title("🌱 Option B — NDVI-Based Potato Yield Forecasting (Multi-County)")

st.sidebar.header("Upload Cleaned Data & Model")
cleaned_file = st.sidebar.file_uploader("Upload ndvi_optionb_cleaned.csv", type=["csv"])
model_file = st.sidebar.file_uploader("Upload rf_optionb_ndvi_model.joblib", type=["joblib"])

if cleaned_file is None or model_file is None:
    st.info("Please upload both cleaned CSV and model to continue.")
    st.stop()

df = pd.read_csv(cleaned_file)
rf = joblib.load(model_file)

# County selector
counties = sorted(df["admin_1"].dropna().unique())
county = st.sidebar.selectbox("County", counties)

# Filter dataset
cdf = df[df["admin_1"] == county].copy()
cdf = cdf.sort_values("harvest_year")

st.header(f"County Selected: {county}")
st.dataframe(cdf.head())

# EDA Plot
st.subheader("Yield over time")
grp = cdf.groupby("harvest_year")["yield"].mean()
fig, ax = plt.subplots()
ax.plot(grp.index, grp.values, marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Yield (t/ha)")
st.pyplot(fig)

# Prepare aggregated dataset
agg = cdf.groupby("harvest_year").agg({
    "area":"sum",
    "production":"sum",
    "mean_annual_ndvi":"mean"
}).reset_index()
agg["yield"] = agg["production"] / agg["area"]

# Multi-year forecasting
st.header("📅 Multi-year Forecasting")

last_year = int(agg["harvest_year"].max())
current_year = datetime.datetime.now().year

target_year = st.number_input(
    "Forecast up to year",
    min_value=last_year + 1,
    max_value=current_year,
    value=last_year + 1,
    step=1
)

def recursive_forecast(df, rf_model, final_year):
    df_hist = df.copy()
    results = []

    for year in range(int(df_hist["harvest_year"].max()) + 1, final_year + 1):
        mean_ndvi = df_hist["mean_annual_ndvi"].iloc[-1]
        area = df_hist["area"].iloc[-1]

        X = np.array([[mean_ndvi, area, 1]])  # planting_month=1
        pred_yield = rf_model.predict(X)[0]
        pred_production = pred_yield * area

        results.append({
            "year": year,
            "predicted_yield": pred_yield,
            "predicted_production": pred_production
        })

        # Append to history
        df_hist.loc[len(df_hist)] = [year, area, pred_production, mean_ndvi, pred_yield]

    return pd.DataFrame(results)

forecast_df = recursive_forecast(agg, rf, int(target_year))
st.subheader("Forecasted Results")
st.dataframe(forecast_df)

final_row = forecast_df.iloc[-1]

# Storage Engine
st.header("❄ Storage Requirement")

def pack_storage(tonnes, sizes=[1000,500,250], fill=0.9):
    required = int(np.ceil(tonnes / fill))
    allocation = []
    remaining = required

    for size in sizes:
        count = remaining // size
        if count > 0:
            allocation.append({"size":size, "count":count})
            remaining -= size * count

    if remaining > 0:
        allocation.append({"size":250, "count":1})

    total = sum(a["size"]*a["count"] for a in allocation)
    util = tonnes / total

    return {"required_capacity":required,"allocation":allocation,"utilization":util}

storage = pack_storage(final_row["predicted_production"])
st.json(storage)

st.download_button(
    "Download Results",
    data=str({"forecast":forecast_df.to_dict(),"storage":storage}),
    file_name="optionb_forecast_output.json"
)
