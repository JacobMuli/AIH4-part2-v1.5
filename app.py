import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
from pathlib import Path

# Paths (change if you saved elsewhere)
CLEAN_CSV = "/mnt/data/ndvi_optionb_cleaned.csv"
MODEL_PATH = "/mnt/data/rf_optionb_ndvi_model.joblib"

st.set_page_config(page_title="Option B — NDVI Yield (Counties)", layout="wide")
st.title("Option B — NDVI-enabled Potato Yield & Storage (Counties)")

# Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_CSV)
    df.columns = df.columns.str.lower().str.strip()
    return df

@st.cache_resource
def load_model():
    if Path(MODEL_PATH).exists():
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
rf = load_model()

if rf is None:
    st.error("Model not found. Run scripts/clean_and_train.py first to create the model.")
    st.stop()

st.sidebar.header("Controls")
selected_county = st.sidebar.selectbox("Select county (admin_1)", sorted(df["admin_1"].dropna().unique()))
min_year = int(df["harvest_year"].min())
max_year = int(df["harvest_year"].max())
st.sidebar.write(f"Data range: {min_year} — {max_year}")

# Forecast target selection (next-year..current year)
last_year = df["harvest_year"].max()
current_year = datetime.datetime.now().year
target_year = st.sidebar.number_input("Forecast up to year", min_value=int(last_year)+1, max_value=current_year, value=int(last_year)+1, step=1)

st.header("Data preview & county selection")
st.write(f"Selected county: **{selected_county}**")
county_df = df[df["admin_1"] == selected_county].sort_values("harvest_year")
st.dataframe(county_df.head(20))

# EDA plots
st.subheader("Yield trend (selected county)")
if "yield" in county_df.columns:
    yr = county_df.groupby("harvest_year")["yield"].mean()
    fig, ax = plt.subplots()
    ax.plot(yr.index, yr.values, marker='o')
    ax.set_xlabel("Harvest Year")
    ax.set_ylabel("Yield (t/ha)")
    st.pyplot(fig)

st.subheader("NDVI vs Yield scatter")
if "mean_annual_ndvi" in county_df.columns:
    fig, ax = plt.subplots()
    ax.scatter(county_df["mean_annual_ndvi"], county_df["yield"])
    ax.set_xlabel("Mean annual NDVI")
    ax.set_ylabel("Yield (t/ha)")
    st.pyplot(fig)

# Build features for county-level aggregated modeling
st.header("Forecast & Storage")

# Aggregate county-level by year (sum area, weighted yield)
agg = county_df.groupby("harvest_year").agg({"area":"sum","production":"sum","mean_annual_ndvi":"mean"}).reset_index()
agg["yield"] = agg["production"] / agg["area"]
agg = agg.sort_values("harvest_year").reset_index(drop=True)
st.write("Aggregated timeseries (county-level):")
st.dataframe(agg)

# Build DF with features for recursive forecasting
def recursive_forecast(grouped_df, rf_model, final_year):
    df_hist = grouped_df.copy().reset_index(drop=True)
    last = int(df_hist["harvest_year"].iloc[-1])
    results = []
    for year in range(last+1, final_year+1):
        # simplest features: use mean_annual_ndvi (last), area (last), planting_month median
        lag1 = df_hist["yield"].iloc[-1]
        lag2 = df_hist["yield"].iloc[-2] if len(df_hist)>=2 else lag1
        lag3 = df_hist["yield"].iloc[-3] if len(df_hist)>=3 else lag1
        roll3 = df_hist["yield"].iloc[-3:].mean() if len(df_hist)>=3 else lag1

        mean_ndvi = df_hist["mean_annual_ndvi"].iloc[-1]
        area = df_hist["area"].iloc[-1]
        # build X similar to training; if model expects county dummies, they were included when training.
        # Here we construct numeric X and append zeros for county dummies if present in model training.
        # Easiest robust approach: use training columns from model.feature_names_in_ if present.
        try:
            feature_names = rf.feature_names_in_
        except Exception:
            feature_names = None

        # Construct base numeric vector
        base = {"mean_annual_ndvi": mean_ndvi, "area": area, "planting_month": 1}
        X_row = pd.DataFrame([base])

        # If model needs adm1 dummies, add zeros (model.feature_names_in_ gives names)
        if feature_names is not None:
            for fn in feature_names:
                if fn not in X_row.columns:
                    X_row[fn] = 0.0
            X_row = X_row[feature_names]
        else:
            # Only numeric features; ensure same order
            X_row = X_row[["mean_annual_ndvi","area","planting_month"]]

        pred_yield = rf_model.predict(X_row.values)[0]
        pred_tonnes = pred_yield * area

        results.append({"year":year,"predicted_yield_t_ha":float(pred_yield),"predicted_tonnage":float(pred_tonnes)})

        # append to history for next loop
        df_hist = pd.concat([df_hist, pd.DataFrame({"harvest_year":[year],"area":[area],"production":[pred_tonnes],"mean_annual_ndvi":[mean_ndvi],"yield":[pred_yield]})], ignore_index=True)
    return pd.DataFrame(results)

# Prepare grouped used for forecasting
if agg.shape[0] < 2:
    st.warning("Not enough historical rows for reliable recursive forecasting for this county.")
else:
    forecast_df = recursive_forecast(agg, rf, int(target_year))
    st.subheader("Forecast results")
    st.dataframe(forecast_df)

    final_row = forecast_df.iloc[-1]
    st.metric("Forecast year", int(final_row["year"]))
    st.metric("Predicted yield (t/ha)", f"{final_row['predicted_yield_t_ha']:.2f}")
    st.metric("Predicted production (tonnes)", f"{final_row['predicted_tonnage']:.0f}")

    # Storage packing (same greedy algorithm)
    def pack_storage(total_tonnes, chamber_sizes=[1000,500,250], max_fill_rate=0.9):
        required_capacity = int(np.ceil(total_tonnes / max_fill_rate))
        allocation=[]
        remaining=required_capacity
        for size in chamber_sizes:
            count = remaining//size
            if count>0:
                allocation.append({"size":size,"count":int(count)})
                remaining -= count*size
        if remaining>0:
            allocation.append({"size":chamber_sizes[-1],"count":1})
        total_alloc = sum([a["size"]*a["count"] for a in allocation])
        utilization = final_row["predicted_tonnage"]/total_alloc
        return {"predicted_tonnes":final_row["predicted_tonnage"], "required_capacity": required_capacity, "allocation": allocation, "total_allocated": total_alloc, "utilization":utilization}

    plan = pack_storage(final_row["predicted_tonnage"])
    st.subheader("Storage plan")
    st.json(plan)

    # Download button
    results = {"forecast_year":int(final_row["year"]),
               "predicted_yield_t_ha":float(final_row["predicted_yield_t_ha"]),
               "predicted_tonnage":float(final_row["predicted_tonnage"]),
               "storage_plan": plan}
    st.download_button("Download forecast & plan (JSON)", data=str(results), file_name="optionb_forecast_plan.json", mime="application/json")
