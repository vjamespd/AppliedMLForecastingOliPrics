# Oil Price Forecasting Web App

This project now includes a simple Streamlit deployment for the oil price forecasting model.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## What the app does

- deploys the best-performing model from the notebook: `ANN + oil`
- retrains the ANN on the cleaned weekly oil price dataset
- forecasts future weekly oil prices using a 4-week lag input window
- shows the saved historical backtest results and model comparison table
- supports a small what-if scenario mode by editing the last four weekly prices
