<!-- COVER IMAGE -->
<div align="center">
  <img src="https://raw.githubusercontent.com/AKilalours/Crypto-Risk-Bench/main/assets/cover.png" alt="Crypto Risk Bench Cover" width="100%"/>

  <h1>📊 Crypto Risk Bench</h1>
  <p><strong>Market Analytics · Risk Metrics · Walk-Forward Backtesting · HTML Dashboard</strong></p>

  <a href="https://github.com/AKilalours/Crypto-Risk-Bench"><img src="https://img.shields.io/badge/repo-GitHub-black?logo=github"/></a>
  <a href="https://drive.google.com/file/d/1IEYqpWRuuxXetZYx_teh0lfwo4m8aD1X/view?usp=sharing"><img src="https://img.shields.io/badge/demo-video-red?logo=google-drive"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker"/>
  <img src="https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions"/>
</div>

---

## ⚡ Performance Snapshot

| Metric | Value |
|---|---|
| **p95 API latency** (cached) | ~180 ms |
| **p95 API latency** (cold run) | ~2.1 s |
| **Cost per request** | ~$0.00 (yfinance, open data) |
| **Backtest MAPE — AAPL (best: `ma7`)** | **3.34%** |
| **Backtest MAPE — BTC-USD (best: `naive`)** | **3.37%** |
| **Cache hit rate** (TTL 1h) | ~85% repeated runs |
| **Latency reduction via caching** | **~91% p95 reduction** (2.1s → 0.18s) |

> Scale note: MAE/RMSE are price-scale-dependent (BTC >> AAPL). For cross-asset comparisons, MAPE is the correct lens.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
│                                                                  │
│  [Yahoo Finance]                                                 │
│       │ yfinance pull (daily OHLCV + intraday 5m)                │
│       ▼                                                          │
│  [TTL Cache Layer]  ←─ .cache_market/ (1hr TTL)                  │
│       │  cache hit → skip fetch                                  │
│       ▼                                                          │
│  [Ingestion + Validation]                                        │
│       │  OHLCV schema check, dedup, fill gaps                    │
│       ▼                                                          │
│  [Feature Engineering]                                           │
│       │  log returns, rolling vol, EWMA, drawdown series         │
│       ▼                                                          │
│  [Risk Analytics]                                                │
│       │  CAGR, Sharpe, Sortino, VaR/CVaR, Max DD, Calmar         │
│       ▼                                                          │
│  [Walk-Forward Backtesting]                                      │
│       │  naive / ma7 / ewma_ret / gbm baselines                  │
│       │  slide window: fit on [0..t], forecast [t..t+H]          │
│       │  aggregate MAE / MSE / RMSE / MAPE                       │
│       ▼                                                          │
│  [Artifact Generation]                                           │
│       │  Plotly HTML dashboards, CSV/JSON reports                │
│       ▼                                                          │
│  [FastAPI Serving]  ──────────────────────────────────────────►  │
│       │  GET /metrics  POST /run  Static /artifacts              │
│       ▼                                                          │
│  [Client / Browser]                                              │
│       artifacts/index.html — browsable dashboard                 │
└──────────────────────────────────────────────────────────────────┘
```

**Trade-offs made:**
| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Latency vs freshness | TTL cache (1hr) | No cache | Eliminates ~2s fetch on repeat runs |
| Interpretability vs accuracy | Baseline models | ML/DL | Evaluation harness is the signal, not alpha |
| Static HTML vs live DB | Plotly HTML | Dashboard DB | Zero infra cost, portable, recruiter-friendly |
| Single process vs queue | CLI + FastAPI | Celery + Redis | Sufficient for demo/portfolio scope |

---

## 🎯 Goals & SLOs

| Goal | SLO / Target |
|---|---|
| Pipeline runs end-to-end without error | 100% on valid data |
| p95 latency (cached API call) | < 300 ms |
| p95 latency (cold run) | < 5 s |
| MAPE on AAPL (best baseline) | < 5% |
| MAPE on BTC-USD (best baseline) | < 6% |
| Artifact generation (HTML/CSV/JSON) | Every run, deterministic |

---

## 🔧 Ops Stack

```
Docker          → containerised pipeline + API server
GitHub Actions  → CI: lint, test, artifact smoke-test on push
FastAPI         → artifact serving + /run + /metrics endpoints
Parquet/CSV     → machine-readable outputs for downstream use
TTL file cache  → reliability layer on top of yfinance
```

**Docker — run the full pipeline in one command:**
```bash
docker build -t crypto-risk-bench .
docker run -p 8000:8000 crypto-risk-bench
# Dashboard → http://localhost:8000/artifacts/index.html
```

**CI (GitHub Actions) — `.github/workflows/ci.yml`:**
```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/
      - run: python main.py --tickers AAPL --days 60   # smoke test
      - run: ls artifacts/reports/metrics.json          # artifact check
```

---

## 🔴 Postmortem — What Broke & How It Was Fixed

### Incident 1 — yfinance flaky downloads (data gaps)
**What broke:** On consecutive runs, `yfinance` occasionally returned empty DataFrames or partial OHLCV data due to upstream throttling or ticker symbol changes (e.g. `BTC-USD` formatting). This caused downstream division-by-zero in the MAPE calculation and NaN risk metrics.

**Fix:**
- Added a **TTL-based file cache** (`.cache_market/`) — repeated runs skip the network call entirely.
- Added schema validation post-download: if shape is 0 or required columns are missing, the pipeline raises a clear `DataIngestionError` rather than silently producing bad artifacts.
- Added a `--refresh-cache` flag for intentional invalidation.

**Result:** Cache hit rate ~85% on re-runs; cold-start now has explicit error messages instead of silent NaNs.

---

### Incident 2 — Walk-forward window off-by-one (data leakage)
**What broke:** An early version of the backtesting loop used `df.iloc[:t+1]` for fitting but included the target day's return in the feature window, causing look-ahead bias and over-optimistic MAPE numbers.

**Fix:**
- Rewrote the window slicing to strictly fit on `df.iloc[:t]` and forecast `df.iloc[t:t+H]`, with a unit test asserting the forecast start index > the last training index.
- Added assertions to `backtest.py` to verify no overlap between train and test windows.

**Result:** MAPE for `naive` on BTC-USD corrected from ~1.8% to ~3.37% — a more honest number.

---

### Incident 3 — HTML dashboard crash on single-ticker runs
**What broke:** The cross-asset correlation heatmap and rolling correlation plots assumed at least 2 tickers were present. Running `--tickers AAPL` caused a KeyError and left `artifacts/index.html` partially generated.

**Fix:**
- Added a guard: cross-asset plots are skipped and a placeholder inserted when fewer than 2 tickers are loaded.
- Dashboard generation now uses a two-pass approach: first validate all data, then render all plots — so index.html is only written if all plots succeed.

**Result:** Single-ticker runs produce a valid, complete dashboard.

---

## 🛡️ Reliability Patterns

| Pattern | Implementation |
|---|---|
| **Caching** | TTL file cache in `.cache_market/` — reduces p95 latency from 2.1s → 0.18s |
| **Fallbacks** | If intraday data unavailable, pipeline continues with daily-only artifacts |
| **Observability** | Structured log lines with timestamps, ticker, and step labels |
| **Graceful degradation** | Cross-asset plots skipped if fewer than 2 tickers; dashboard still generated |
| **Retries** | `--refresh-cache` flag forces re-fetch without modifying other state |

---

## 📐 Why This Project Is Credible

Many student forecasting repos stop at a plot. Crypto Risk Bench focuses on the layers recruiters actually assess:

- **Evaluation discipline** — walk-forward backtesting, not a single lucky split
- **Risk analytics** — drawdown, Sharpe/Sortino, VaR/CVaR, Calmar
- **Artifact-first workflow** — HTML dashboards + CSV/JSON reports every run
- **Ops awareness** — Docker, CI/CD, TTL caching, structured logs
- **Postmortem culture** — documented failures and fixes, not just successes
- **Demo-friendly** — FastAPI server, Swagger UI, live `/metrics` endpoint

---

## 📦 What It Produces

### Risk & performance summary (per asset)
- CAGR (annualized return)
- Annualized volatility
- Sharpe & Sortino ratios
- Max drawdown + Calmar ratio
- VaR/CVaR on log returns
- Skewness / kurtosis of returns

### Baseline forecasting + walk-forward backtesting
Interpretable baselines evaluated with walk-forward backtesting:

| Model | Description |
|---|---|
| `naive` | Last observed value |
| `ma7` | 7-day moving-average level |
| `ewma_ret` | EWMA on log returns, compounded forward |
| `gbm` | GBM-style simulation with 10–90% confidence band |

**Best baseline per ticker (lowest RMSE):**

| Ticker | Best Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| AAPL | `ma7` | 7.97 | 10.37 | **3.34%** |
| BTC-USD | `naive` | 3,535.82 | 4,906.83 | **3.37%** |

> MAE/RMSE are scale-dependent. Prefer **MAPE** for cross-asset comparison.

### Dashboards & artifacts
- Plotly HTML dashboards per ticker
- Cross-asset normalized performance comparison (base = 100)
- Returns correlation heatmap + rolling correlation
- Machine-readable `metrics.json`

---

## 🚀 Quickstart

### Requirements
- Python 3.10+ (tested on 3.12)
- Internet access (data via `yfinance`)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run pipeline
```bash
python main.py --open-index
```
Then open: `artifacts/index.html`

---

## 🐳 Docker

```bash
docker build -t crypto-risk-bench .
docker run -p 8000:8000 crypto-risk-bench
```

---

## 🌐 FastAPI Demo

```bash
pip install fastapi uvicorn
python main.py --serve-api --api-port 8000
```

| Endpoint | Description |
|---|---|
| `GET /artifacts/index.html` | Browsable dashboard |
| `GET /health` | Health check |
| `GET /metrics` | Latest `metrics.json` |
| `POST /run` | Trigger a pipeline run |
| `GET /docs` | Swagger UI |

---

## 🖥️ CLI Examples

```bash
python main.py --tickers BTC-USD AAPL --days 365
python main.py --plot-last-days 180
python main.py --forecast-days 15
python main.py --backtest-horizon 15 --backtest-steps 8
python main.py --refresh-cache
```

---

## 📁 Output Structure

```
artifacts/
├── index.html
├── plots/
│   ├── <ticker>_dashboard.html
│   ├── <ticker>_ohlc.html
│   ├── <ticker>_forecast_gbm.html
│   ├── <ticker>_intraday.html
│   ├── normalized_performance.html
│   ├── returns_correlation_heatmap.html
│   └── rolling_correlation_30d_<a>_vs_<b>.html
├── reports/
│   ├── summary.html
│   ├── summary.md
│   ├── performance_risk_summary.csv
│   ├── backtest_model_comparison.csv
│   └── metrics.json
└── data/
    ├── <ticker>_daily.csv
    └── <ticker>_daily.parquet
```

---

## 🗂️ Key Files for Evaluation

| File | Contents |
|---|---|
| `artifacts/reports/backtest_model_comparison.csv` | MAE/MSE/RMSE/MAPE per model per ticker |
| `artifacts/reports/performance_risk_summary.csv` | CAGR, Sharpe, Sortino, VaR/CVaR, Max DD |
| `artifacts/reports/metrics.json` | Machine-readable full metrics |
| `artifacts/index.html` | Browsable dashboard entry point |

---

## ⚙️ Tech Stack

`Python` · `pandas` · `numpy` · `yfinance` · `Plotly` · `FastAPI` · `Uvicorn` · `Docker` · `GitHub Actions` · `Parquet` · `CSV`

---

## ⚠️ Limitations

- Uses `yfinance` — availability and rate limits may vary
- Forecast models are interpretable baselines, not trading/alpha claims
- No trading strategy, transaction costs, or execution simulation included
- Intraday availability depends on ticker and Yahoo Finance constraints
