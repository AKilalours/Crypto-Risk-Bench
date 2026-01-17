# CryptoRiskLab — Market Analytics, Risk Metrics, Walk-Forward Backtesting & Dashboard (BTC-USD, AAPL)

CryptoRiskLab is a production-style **market analytics + risk + evaluation** pipeline that ingests daily and intraday price data, engineers return/risk features, runs **walk-forward backtests** across interpretable baseline forecasters, and generates a browsable **HTML dashboard** (plots + reports). It also includes an optional **FastAPI** service for demos and artifact serving.

**Demo video:** https://drive.google.com/file/d/1IEYqpWRuuxXetZYx_teh0lfwo4m8aD1X/view?usp=sharing

> This repository is intentionally positioned as **analytics + evaluation + reproducible reporting**, not “crypto price prediction for profit.”  
> Outputs include standard evaluation metrics (MAE/RMSE/MAPE) and risk statistics (CAGR, Sharpe/Sortino, Max Drawdown, VaR/CVaR).

---

## Why this project is credible
Many student “forecasting” repositories stop at a plot. CryptoRiskLab focuses on the engineering and evaluation layers recruiters actually assess:

- **Reproducible runs**: CLI configuration + deterministic seed
- **Caching & reliability**: TTL-based caching to reduce flaky upstream downloads
- **Evaluation discipline**: walk-forward backtesting (not a single random split)
- **Risk analytics**: drawdown, volatility, Sharpe/Sortino, VaR/CVaR
- **Artifact-first workflow**: HTML dashboards + CSV/JSON reports generated every run
- **Demo-friendly**:  FastAPI service to serve artifacts and trigger runs

---

## What it produces

### Risk & performance summary (per asset)
- CAGR (annualized return)
- Annualized volatility
- Sharpe & Sortino ratios
- Max drawdown + Calmar ratio
- VaR/CVaR on log returns
- Skewness / kurtosis of returns

### Baseline forecasting + walk-forward backtesting (evaluation-first)
Baselines evaluated with **walk-forward** backtesting:
- `naive`: last observed value
- `ma7`: 7-day moving-average level
- `ewma_ret`: EWMA on log returns, compounded forward
- `gbm`: GBM-style simulation baseline with 10–90% band

Metrics written to reports:
- **MAE**, **RMSE**, **MAPE**

> These models are intentionally interpretable baselines. The primary engineering signal is the evaluation harness + reporting.

### Dashboards & artifacts
- Plotly HTML dashboards per ticker
- Cross-asset normalized performance comparison (start = 100)
- Returns correlation heatmap + rolling correlation
- Machine-readable metrics JSON

---
## Data

CryptoRiskLab pulls market data **on demand** from **Yahoo Finance** using the `yfinance` library.

- **Source:** Yahoo Finance (via `yfinance`)
- **Assets (default):** `BTC-USD`, `AAPL`
- **Granularity:**
  - **Daily OHLCV** for the configured date range (`--days`, `--end`)
  - **Intraday** close prices using `period=1d` with configurable interval (default: `5m`)
- **Daily fields (when available):** Open, High, Low, Close, Adj Close, Volume  
- **Intraday fields (when available):** DateTime, Open, High, Low, Close, Volume  

Notes:
- Data availability may vary due to upstream limits, market hours, or delayed feeds.
- TTL-based caching (`.cache_market/`) reduces repeated downloads and improves reliability.

## Evaluation Results (Walk-Forward Backtesting)

Baselines are evaluated with **walk-forward backtesting** and reported using standard error metrics:

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

**Best baseline per ticker (lowest RMSE):**
- **AAPL (best = `ma7`)** — MAE **7.9738**, MSE **107.4526**, RMSE **10.3659**, MAPE **3.3421%**
- **BTC-USD (best = `naive`)** — MAE **3535.8244**, MSE **24,077,030**, RMSE **4906.8348**, MAPE **3.3716%**

Where these values are written:
- `artifacts/reports/backtest_model_comparison.csv`
- `artifacts/reports/metrics.json`

---

## Quickstart

### Requirements
- Python 3.10+ (tested on 3.12)
- Internet access (data via `yfinance`)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
# Run pipeline (generate  artifacts)

python main.py --open-index

Open the dashboard:
  - artifacts/index.html

---

# API demo mode (FastAPI)

- Install API Dependencies

python -m pip install fastapi uvicorn

- Start the server

python main.py --serve-api --api-port 8000

Open:

- Dashboard: http://127.0.0.1:8000/artifacts/index.html
- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

# Trigger a run via API

- POST /run (optional: refresh_cache=true)
- GET /metrics for the latest metrics.json

---

# Output Structure

artifacts/
  index.html
  plots/
    <ticker>_dashboard.html
    <ticker>_ohlc.html
    <ticker>_forecast_gbm.html
    <ticker>_intraday.html
    normalized_performance.html
    returns_correlation_heatmap.html
    rolling_correlation_30d_<a>_vs_<b>.html
  reports/
    summary.html
    summary.md
    performance_risk_summary.csv
    backtest_model_comparison.csv
    metrics.json
  data/
    <ticker>_daily.csv
    <ticker>_daily.parquet   

---
# CLI examples

python main.py --tickers BTC-USD AAPL --days 365
python main.py --plot-last-days 180
python main.py --forecast-days 15
python main.py --backtest-horizon 15 --backtest-steps 8
python main.py --refresh-cache

---

# Where to look (for evaluation)

- artifacts/reports/backtest_model_comparison.csv
- artifacts/reports/performance_risk_summary.csv
- artifacts/reports/metrics.json
- artifacts/index.html

---

# Limitations

•	Uses yfinance (availability and rate limits may vary).
•	Forecast models are baselines, not trading/alpha claims.
•	No trading strategy, transaction costs, or execution simulation included.
---

# Tech stack

Python, pandas, numpy, yfinance, Plotly, FastAPI, Uvicorn, Parquet/CSV caching

---



