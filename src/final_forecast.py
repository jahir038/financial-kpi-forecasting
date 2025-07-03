# final_forecast.py
import pandas as pd
import numpy as np
import joblib

def predict_next_quarter(df_corr, corr_models, predictor_cols=["DL_HC", "IDL_HC"]):

    # Identify latest quarter (e.g., FY25 Q4)
    quarter_order = sorted(df_corr["Quarter"].unique(), key=lambda x: (int(x[2:4]), int(x[-1])))
    latest_quarter = quarter_order[-1]

    results = []

    for kpi in df_corr.columns:
        if kpi in ["MEP", "Quarter"] + predictor_cols:
            continue  # Skip non-KPI columns

        for mep, group in df_corr.groupby("MEP"):
            key = (mep, kpi)
            if key not in corr_models:
                continue

            model = corr_models[key]

            # Get latest input row
            latest_row = group[group["Quarter"] == latest_quarter]
            if latest_row.empty:
                continue

            input_feats = latest_row[predictor_cols].values.reshape(1, -1)
            prediction = model.predict(input_feats)[0]
            coef = model.coef_ if hasattr(model, "coef_") else [np.nan] * len(predictor_cols)

            row_data = {
                "MEP": mep,
                "KPI": kpi,
                "Predicted_FY26_Q1": prediction,
                "DL_HC": latest_row["DL_HC"].values[0],
                "IDL_HC": latest_row["IDL_HC"].values[0]
            }

            for i, col in enumerate(predictor_cols):
                row_data[f"Coef_{col}"] = coef[i]

            # Optionally: include full KPI history
            past_quarters = group.set_index("Quarter").sort_index()
            for qtr in past_quarters.index:
                row_data[qtr] = past_quarters.loc[qtr, kpi]

            results.append(row_data)

    return pd.DataFrame(results)
