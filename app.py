import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
import shap
from lime.lime_tabular import LimeTabularExplainer

# ----------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------
st.set_page_config(page_title="Option A — NDVI Yield Intelligence", layout="wide")
st.title("🌱 Option A — NDVI-Based Potato Yield Forecasting")

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
st.sidebar.header("Upload Cleaned Data & Model")
data_file = st.sidebar.file_uploader("Upload ndvi_optionb_cleaned.csv", type=["csv"])
model_file = st.sidebar.file_uploader("Upload rf_optionb_ndvi_model.joblib", type=["joblib"])

if not data_file or not model_file:
    st.info("Please upload both dataset and trained model.")
    st.stop()

# Load data
df = pd.read_csv(data_file)
rf = joblib.load(model_file)

df.columns = df.columns.str.lower().str.strip()

# ----------------------------------------------------------
# COUNTY & AREA SELECTION
# ----------------------------------------------------------
counties = sorted(df["admin_1"].dropna().unique())
county = st.sidebar.selectbox("Select County", counties)

area_override = st.sidebar.number_input(
    "Enter Area to Forecast (ha)",
    min_value=1.0,
    value=float(df[df["admin_1"] == county]["area"].mean()),
    help="Overrides dataset area for forecasting."
)

# Filter dataset for this county
cdf = df[df["admin_1"] == county].copy().sort_values("harvest_year")


# ----------------------------------------------------------
# GLOBAL FUNCTION: Prepare features for SHAP & LIME
# ----------------------------------------------------------
def prepare_shap_features(df_rows, model):
    X = df_rows.copy()

    # Base feature set
    X_feat = X[["mean_annual_ndvi", "area"]].copy()
    X_feat["area"] = area_override
    X_feat["planting_month"] = 1

    # One-hot encode counties exactly as model expects
    if hasattr(model, "feature_names_in_"):
        for feat in model.feature_names_in_:
            if feat.startswith("adm1_"):
                county_name = feat.replace("adm1_", "")
                X_feat[feat] = (X["admin_1"] == county_name).astype(int)

        X_feat = X_feat[model.feature_names_in_]

    return X_feat


# ----------------------------------------------------------
# PAGE NAVIGATION
# ----------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Forecasting", "SHAP Explainability", "LIME Explainability"]
)


# ----------------------------------------------------------
# FORECASTING PAGE
# ----------------------------------------------------------
if page == "Forecasting":

    st.header(f"County Selected: **{county}**")
    st.write("Dataset Preview:")
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

    # ------------------------------
    # AGGREGATION FOR FORECASTING
    # ------------------------------
    agg = cdf.groupby("harvest_year").agg({
        "area": "sum",
        "production": "sum",
        "mean_annual_ndvi": "mean"
    }).reset_index()

    agg["yield"] = agg["production"] / agg["area"]
    agg["admin_1"] = county

    # ------------------------------
    # FEATURE RECONSTRUCTION
    # ------------------------------
    def build_feature_vector(last_row, rf_model):
        base = {
            "mean_annual_ndvi": float(last_row["mean_annual_ndvi"]),
            "area": float(area_override),
            "planting_month": 1
        }

        if hasattr(rf_model, "feature_names_in_"):
            expected = list(rf_model.feature_names_in_)
            row = {}

            for feat in expected:
                if feat in base:
                    row[feat] = base[feat]
                elif feat.startswith("adm1_"):
                    row[feat] = 1.0 if feat == f"adm1_{county}" else 0.0
                else:
                    row[feat] = 0.0

            return np.array([[row[f] for f in expected]], dtype=float)

        return np.array([[base["mean_annual_ndvi"], base["area"], base["planting_month"]]])

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
        value=last_year + 1
    )

    def recursive_forecast(history_df, rf_model, final_year):
        hist = history_df.copy().reset_index(drop=True)
        results = []

        for year in range(int(hist["harvest_year"].max()) + 1, final_year + 1):
            last = hist.iloc[-1]
            X = build_feature_vector(last, rf_model)

            pred_yield = float(rf_model.predict(X)[0])
            pred_prod = pred_yield * area_override

            results.append({
                "year": year,
                "predicted_yield": pred_yield,
                "predicted_production": pred_prod
            })

            new_row = {
                "harvest_year": year,
                "area": area_override,
                "production": pred_prod,
                "mean_annual_ndvi": last["mean_annual_ndvi"],
                "yield": pred_yield,
                "admin_1": county
            }
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(results)

    forecast_df = recursive_forecast(agg, rf, int(target_year))
    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    final_tonnes = float(forecast_df.iloc[-1]["predicted_production"])

    # ------------------------------
    # STORAGE ENGINE (IMPROVED)
    # ------------------------------
    st.header("❄ Cold Storage Requirement (Type 1, Type 2, Type 3)")

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

        typed_allocation = {
            f"type {i+1}": allocation[i]
            for i in range(len(allocation))
        }

        total_capacity = sum(a["size"] * a["count"] for a in allocation)
        utilization = tonnes / total_capacity if total_capacity else 0

        return {
            "required_capacity": needed,
            "allocation": typed_allocation,
            "utilization": utilization
        }

    storage = pack_storage(final_tonnes)
    st.json(storage)

    st.download_button(
        "Download Forecast Output",
        data=str({"forecast": forecast_df.to_dict(), "storage": storage}),
        file_name="optionA_forecast_output.json"
    )


# ----------------------------------------------------------
# SHAP PAGE
# ----------------------------------------------------------
elif page == "SHAP Explainability":
    st.header("🔍 SHAP Explainability")

    X_shap = prepare_shap_features(cdf, rf)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)

    st.subheader("Feature Importance (Bar Plot)")
    fig = plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("SHAP Dot Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    st.pyplot(fig)


# ----------------------------------------------------------
# LIME PAGE
# ----------------------------------------------------------
elif page == "LIME Explainability":
    st.header("🟩 LIME Explainability")

    X_shap = prepare_shap_features(cdf, rf)
    X_lime = X_shap.values

    lime_expl = LimeTabularExplainer(
        X_lime,
        feature_names=X_shap.columns.tolist(),
        mode="regression"
    )

    idx = st.number_input(
        "Select row index to explain",
        min_value=0,
        max_value=len(X_lime) - 1,
        value=0
    )

    lime_exp = lime_expl.explain_instance(
        X_lime[int(idx)],
        rf.predict,
        num_features=10
    )

    st.subheader("LIME Feature Contributions")
    st.json(lime_exp.as_list())
    st.pyplot(lime_exp.as_pyplot_figure())


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import datetime
# import shap
# from lime.lime_tabular import LimeTabularExplainer

# st.set_page_config(page_title="Option A — NDVI Yield Intelligence", layout="wide")
# st.title("🌱 Option A — NDVI-Based Potato Yield Forecasting")

# # ------------------------------
# # FILE UPLOAD
# # ------------------------------
# st.sidebar.header("Upload Cleaned Data & Model")
# data_file = st.sidebar.file_uploader("Upload ndvi_optionb_cleaned.csv", type=["csv"])
# model_file = st.sidebar.file_uploader("Upload rf_optionb_ndvi_model.joblib", type=["joblib"])

# if not data_file or not model_file:
#     st.info("Please upload cleaned dataset and trained model to continue.")
#     st.stop()

# df = pd.read_csv(data_file)
# rf = joblib.load(model_file)

# df.columns = df.columns.str.lower().str.strip()

# # ------------------------------
# # COUNTY SELECTION
# # ------------------------------
# counties = sorted(df["admin_1"].dropna().unique())
# county = st.sidebar.selectbox("Select County", counties)

# # AREA INPUT OVERRIDE (NEW)
# area_override = st.sidebar.number_input(
#     "Enter Area to Forecast (ha)",
#     min_value=1.0,
#     value=float(df[df["admin_1"] == county]["area"].mean()),
#     step=1.0,
#     help="Overrides area from dataset for forecasting."
# )

# # Filter dataset
# cdf = df[df["admin_1"] == county].copy().sort_values("harvest_year")

# st.header(f"County Selected: **{county}**")
# st.write("Dataset (filtered):")
# st.dataframe(cdf.head())

# # ------------------------------
# # BASIC EDA
# # ------------------------------
# st.subheader("📊 Yield Over Time")

# if "yield" in cdf.columns:
#     by_year = cdf.groupby("harvest_year")["yield"].mean()
#     fig, ax = plt.subplots()
#     ax.plot(by_year.index, by_year.values, marker="o")
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Yield (t/ha)")
#     ax.set_title("Mean Yield per Year")
#     st.pyplot(fig)
# else:
#     st.warning("Column 'yield' missing from dataset.")

# # ------------------------------
# # AGGREGATION FOR FORECASTING
# # ------------------------------
# agg = cdf.groupby("harvest_year").agg({
#     "area": "sum",
#     "production": "sum",
#     "mean_annual_ndvi": "mean"
# }).reset_index()

# agg["yield"] = agg["production"] / agg["area"]
# agg["admin_1"] = county  # Required for SHAP/LIME + feature reconstruction

# # ------------------------------
# # FEATURE RECONSTRUCTION FOR MODEL
# # ------------------------------
# def build_feature_vector(last_row, rf_model):
#     base = {
#         "mean_annual_ndvi": float(last_row.get("mean_annual_ndvi", 0.0)),
#         "area": float(area_override),  # USE OVERRIDE AREA (NEW)
#         "planting_month": 1
#     }

#     if hasattr(rf_model, "feature_names_in_"):
#         expected = list(rf_model.feature_names_in_)
#         row = {}

#         for feat in expected:
#             if feat in base:
#                 row[feat] = base[feat]
#             elif feat.startswith("adm1_"):
#                 row[feat] = 1.0 if feat == f"adm1_{last_row.get('admin_1','')}" else 0.0
#             else:
#                 row[feat] = 0.0

#         X = np.array([[row[f] for f in expected]], dtype=float)
#         return X

#     return np.array([[base["mean_annual_ndvi"], base["area"], base["planting_month"]]], dtype=float)

# # ------------------------------
# # MULTI-YEAR FORECASTING
# # ------------------------------
# st.header("📅 Multi-Year Forecasting")

# last_year = int(agg["harvest_year"].max())
# current_year = datetime.datetime.now().year

# target_year = st.number_input(
#     "Forecast up to year:",
#     min_value=last_year + 1,
#     max_value=current_year,
#     value=last_year + 1,
# )

# def recursive_forecast(history_df, rf_model, final_year):
#     hist = history_df.copy().reset_index(drop=True)
#     results = []

#     for year in range(int(hist["harvest_year"].max()) + 1, final_year + 1):
#         last = hist.iloc[-1]

#         X = build_feature_vector(last, rf_model)

#         pred_yield = float(rf_model.predict(X)[0])
#         pred_prod = pred_yield * area_override  # USE OVERRIDDEN AREA

#         results.append({
#             "year": year,
#             "predicted_yield": pred_yield,
#             "predicted_production": pred_prod
#         })

#         new_row = {
#             "harvest_year": year,
#             "area": area_override,
#             "production": pred_prod,
#             "mean_annual_ndvi": last["mean_annual_ndvi"],
#             "yield": pred_yield,
#             "admin_1": last.get("admin_1")
#         }
#         hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

#     return pd.DataFrame(results)

# forecast_df = recursive_forecast(agg, rf, int(target_year))

# st.subheader("Forecast Results")
# st.dataframe(forecast_df)

# final_tonnes = float(forecast_df.iloc[-1]["predicted_production"])

# # ------------------------------
# # STORAGE PLANNING
# # ------------------------------
# st.header("❄ Cold Storage Requirement")

# def pack_storage(tonnes, sizes=[1000, 500, 250], fill=0.9):
#     needed = int(np.ceil(tonnes / fill))
#     allocation = []
#     remaining = needed

#     for s in sizes:
#         c = remaining // s
#         if c > 0:
#             allocation.append({"size": s, "count": c})
#             remaining -= c * s

#     if remaining > 0:
#         allocation.append({"size": 250, "count": 1})

#     total = sum(x["size"] * x["count"] for x in allocation)
#     util = tonnes / total
#     return {"required_capacity": needed, "allocation": allocation, "utilization": util}

# storage = pack_storage(final_tonnes)
# st.json(storage)

# # ------------------------------
# # DOWNLOAD OUTPUT
# # ------------------------------
# st.download_button(
#     "Download Forecast Output",
#     data=str({"forecast": forecast_df.to_dict(), "storage": storage}),
#     file_name="optionA_forecast_output.json"
# )

# # ============================================================
# #                  SHAP EXPLAINABILITY PAGE (NEW)
# # ============================================================
# st.header("🔍 SHAP Explainability")

# # Build feature matrix for SHAP
# def prepare_shap_features(df, model):
#     X = df.copy()

#     # Base numeric features
#     X_feat = X[["mean_annual_ndvi", "area"]].copy()
#     X_feat["planting_month"] = 1

#     # Add county dummy columns if model used them
#     if hasattr(model, "feature_names_in_"):
#         for feat in model.feature_names_in_:
#             if feat.startswith("adm1_"):
#                 X_feat[feat] = (X["admin_1"] == feat.replace("adm1_", "")).astype(int)

#     # Order features to match model
#     if hasattr(model, "feature_names_in_"):
#         X_feat = X_feat[model.feature_names_in_]

#     return X_feat

# X_shap = prepare_shap_features(cdf, rf)

# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_shap)

# st.subheader("Feature Importance (SHAP Summary Bar)")
# fig = plt.figure()
# shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
# st.pyplot(fig)

# st.subheader("SHAP Dot Plot")
# fig = plt.figure()
# shap.summary_plot(shap_values, X_shap, show=False)
# st.pyplot(fig)

# # ============================================================
# #                  LIME EXPLAINABILITY PAGE (NEW)
# # ============================================================
# st.header("🟩 LIME Explainability")

# X_lime = X_shap.values
# lime_explainer = LimeTabularExplainer(
#     X_lime,
#     feature_names=X_shap.columns.tolist(),
#     mode="regression"
# )

# idx = st.number_input(
#     "Select row index to explain:",
#     min_value=0,
#     max_value=len(X_shap) - 1,
#     value=0
# )

# lime_exp = lime_explainer.explain_instance(
#     X_lime[int(idx)],
#     rf.predict,
#     num_features=10
# )

# st.subheader("LIME Feature Contributions")
# st.json(lime_exp.as_list())

# fig = lime_exp.as_pyplot_figure()
# st.pyplot(fig)
