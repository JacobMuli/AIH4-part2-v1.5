🌱 Option B — NDVI-Based Yield Intelligence & Cold Storage Planning

A machine-learning system for multi-county potato yield forecasting using a cleaned NDVI-enhanced dataset.
This repository includes:

Data cleaning pipeline

RandomForest yield prediction model

Multi-year recursive forecasting

Multi-county analytics

Cold storage recommendation engine

Streamlit dashboard

Google Colab workflow for model training

This solution runs entirely from the uploaded NDVI dataset and does not require external satellite or weather APIs.

📦 Key Features
✔ NDVI-Enhanced Yield Modeling

Uses mean_annual_ndvi, area, planting month, and county metadata to predict potato yields across Kenya.

✔ Multi-County Support

The model works for all counties in the dataset (admin_1), not Meru only.

✔ Recursive Multi-Year Forecasting

Forecast yield and production year by year, up to the current year.

✔ Storage Requirement Engine

Recommends cold storage chamber configurations (1000t, 500t, 250t) based on predicted production.

✔ Streamlit Dashboard

Interactive app allows users to:

Upload cleaned CSV + model

Choose county

View NDVI–yield trends

View forecasted yields

Download storage plan JSON

✔ Colab-Friendly Training Flow

Train, clean, export model artifacts, and download outputs directly from Google Colab.

📁 Repository Structure
option-b-yield-intelligence/
│
├── app.py                           # Streamlit dashboard (upload cleaned CSV + model)
├── requirements.txt                 # Python dependencies
│
├── scripts/
│   └── clean_and_train.ipynb        # Google Colab-ready training notebook (recommended)
│   └── clean_and_train.py           # CLI Python script (optional)
│
├── data/
│   └── ndvi_filled_option_c_poly.xlsx   # Raw NDVI dataset (not committed publicly)
│
├── models/
│   └── rf_optionb_ndvi_model.joblib     # Exported model (from Colab)
│
├── outputs/
│   ├── ndvi_optionb_cleaned.csv         # Clean dataset (after agronomic year repair)
│   └── optionb_results_summary.json     # Evaluation metrics + metadata
│
└── README.md                           # This document

📊 Data Schema (Cleaned)
Column	Description
fnid	Field identifier
country	Kenya
admin_1	County
admin_2	Sub-county
product	Crop type (Potato)
season_name	Season label
planting_year	Corrected planting year
planting_month	Planting month
harvest_year	Corrected harvest year (fixed if originally wrong)
harvest_month	Harvest month
area	Cultivated area (ha)
production	Production (tonnes)
yield	Yield (t/ha)
mean_annual_ndvi	NDVI predictor value
🧹 Data Cleaning Logic

The uploaded dataset contained agronomic inconsistencies, such as:

planting_year = 2017
harvest_year = 2016  (impossible)


The cleaning script:

Detects any row where planting_year > harvest_year

Fixes it by setting:

harvest_year = planting_year


Outputs a cleaned file:

outputs/ndvi_optionb_cleaned.csv

🖥 Streamlit Dashboard

The Streamlit app (app.py) allows:

Uploading cleaned CSV + model

Selecting county

Viewing NDVI/yield relationships

Multi-year recursive forecasting

Cold storage estimation

JSON download

Run locally:

streamlit run app.py

❄ Storage Engine Algorithm

Storage is allocated using a greedy approach:

Chamber sizes: 1000t, 500t, 250t

Fill rate: 90%

Output:

Required total capacity

Optimal chamber mix

Utilization rate

🔮 Future Enhancements

XGBoost and LightGBM versions

SHAP explainability

FPO-level aggregation

Pricing and market intelligence

Weather integration (Option C)

📬 Support

For professional engineering support or system design:
👉 jacobmwalughs@gmail.com
