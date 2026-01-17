from __future__ import annotations

import argparse
import dataclasses
import html
import json
import logging
import math
import os
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Logging
# ============================================================
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("market_analysis_adv")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
    return logger


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class AnalysisConfig:
    tickers: Tuple[str, ...]
    start: date
    end: date

    intraday_interval: str = "5m"
    cache_ttl_hours: int = 12
    intraday_cache_ttl_minutes: int = 45
    refresh_cache: bool = False

    outdir: Path = Path("artifacts")
    cache_dir: Path = Path(".cache_market")
    seed: int = 42

    # Backtesting / forecasting
    forecast_days: int = 15
    backtest_horizon: int = 15
    backtest_steps: int = 8
    models: Tuple[str, ...] = ("naive", "ma7", "ewma_ret", "gbm")

    # Plot controls
    plot_last_days: int = 180
    open_index: bool = False


# ============================================================
# Small utilities
# ============================================================
def safe_name(s: str) -> str:
    """Filename-safe: keep alnum, dash, underscore; replace others with '_'."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


def to_jsonable(x):
    """Recursively convert dates/Paths/numpy types to JSON-serializable values."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (date, datetime, Path)):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)


# ============================================================
# Cache utils (parquet preferred, csv fallback)
# ============================================================
def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance can return MultiIndex columns like ('Close', 'AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def _is_cache_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


def _cache_path(cache_dir: Path, key: str, suffix: str = ".parquet") -> Path:
    return cache_dir / f"{safe_name(key)}{suffix}"


def _safe_read_cache(parquet_path: Path, csv_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Cache: failed to read parquet ({parquet_path.name}): {e}")

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"Cache: failed to read csv ({csv_path.name}): {e}")

    return None


def _safe_write_cache(df: pd.DataFrame, parquet_path: Path, csv_path: Path, logger: logging.Logger) -> None:
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Cache saved: {parquet_path}")
        return
    except Exception as e:
        logger.warning(f"Cache: parquet write failed; falling back to csv. ({e})")

    df.to_csv(csv_path, index=False)
    logger.info(f"Cache saved: {csv_path}")


# ============================================================
# Data acquisition
# ============================================================
def download_history(
    ticker: str,
    start: date,
    end: date,
    logger: logging.Logger,
    cache_dir: Path,
    cache_ttl_hours: int,
    refresh_cache: bool,
    max_attempts: int = 4,
    sleep_s: float = 1.2,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"daily_{ticker}_{start.isoformat()}_{end.isoformat()}"
    cp_parquet = _cache_path(cache_dir, key, ".parquet")
    cp_csv = _cache_path(cache_dir, key, ".csv")

    ttl = cache_ttl_hours * 3600
    if (not refresh_cache) and (_is_cache_fresh(cp_parquet, ttl) or _is_cache_fresh(cp_csv, ttl)):
        cached = _safe_read_cache(cp_parquet, cp_csv, logger)
        if cached is not None and not cached.empty:
            cached = _flatten_yfinance_columns(cached)
            if "Date" in cached.columns:
                cached["Date"] = pd.to_datetime(cached["Date"]).dt.tz_localize(None)
            return cached

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"{ticker}: downloading daily history (attempt {attempt}/{max_attempts})")
            df = yf.download(
                ticker,
                start=start,
                end=end + timedelta(days=1),
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if df is None or df.empty:
                raise RuntimeError("Empty daily data returned.")

            df = df.reset_index()
            df = _flatten_yfinance_columns(df)

            df.rename(columns={"index": "Date"}, inplace=True)
            if "Date" not in df.columns:
                raise RuntimeError("Missing Date column after reset_index().")

            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

            cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            keep = [c for c in cols if c in df.columns]
            df = df[keep].copy()
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

            _safe_write_cache(df, cp_parquet, cp_csv, logger)
            return df
        except Exception as e:
            last_err = e
            logger.warning(f"{ticker}: download failed: {e}")
            time.sleep(sleep_s * attempt)

    raise last_err or RuntimeError(f"{ticker}: failed to download daily history.")


def download_intraday(
    ticker: str,
    interval: str,
    logger: logging.Logger,
    cache_dir: Path,
    ttl_minutes: int,
    refresh_cache: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    key = f"intraday_{ticker}_{interval}_{today}"
    cp_parquet = _cache_path(cache_dir, key, ".parquet")
    cp_csv = _cache_path(cache_dir, key, ".csv")

    ttl = ttl_minutes * 60
    if (not refresh_cache) and (_is_cache_fresh(cp_parquet, ttl) or _is_cache_fresh(cp_csv, ttl)):
        cached = _safe_read_cache(cp_parquet, cp_csv, logger)
        if cached is not None and not cached.empty:
            cached = _flatten_yfinance_columns(cached)
            if "DateTime" in cached.columns:
                cached["DateTime"] = pd.to_datetime(cached["DateTime"]).dt.tz_localize(None)
            return cached

    logger.info(f"{ticker}: fetching intraday (period=1d, interval={interval})")
    df = yf.download(
        ticker,
        period="1d",
        interval=interval,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df = _flatten_yfinance_columns(df)

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "DateTime"}, inplace=True)

    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)

    _safe_write_cache(df, cp_parquet, cp_csv, logger)
    return df


# ============================================================
# Feature engineering + risk/performance analytics
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"]).copy()
    out["ret"] = out["Close"].pct_change()
    out["log_return"] = np.log(out["Close"]).diff()
    out["ma_7"] = out["Close"].rolling(7).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["vol_20"] = out["log_return"].rolling(20).std() * math.sqrt(252)
    out["rolling_sharpe_60"] = (
        out["log_return"].rolling(60).mean() / (out["log_return"].rolling(60).std() + 1e-12)
    ) * math.sqrt(252)
    running_max = out["Close"].cummax()
    out["drawdown"] = out["Close"] / running_max - 1.0
    return out


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return float("nan")
    peak = close.cummax()
    dd = close / peak - 1.0
    return float(dd.min())


def _cagr(close: pd.Series, dates: pd.Series) -> float:
    if close.empty:
        return float("nan")
    start_val = float(close.iloc[0])
    end_val = float(close.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return float("nan")
    start_date = pd.to_datetime(dates.iloc[0])
    end_date = pd.to_datetime(dates.iloc[-1])
    years = max((end_date - start_date).days / 365.25, 1e-9)
    return float((end_val / start_val) ** (1 / years) - 1)


def _var_cvar(log_returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    r = log_returns.dropna().to_numpy()
    if r.size == 0:
        return float("nan"), float("nan")
    var = np.quantile(r, alpha)
    cvar = r[r <= var].mean() if np.any(r <= var) else var
    return float(var), float(cvar)


def compute_performance_stats(df: pd.DataFrame) -> Dict[str, float]:
    df = df.dropna(subset=["Date", "Close"]).copy()
    if df.empty:
        return {}

    log_ret = df["log_return"].dropna()
    ann_vol = float(log_ret.std() * math.sqrt(252)) if not log_ret.empty else float("nan")
    ann_ret = _cagr(df["Close"], df["Date"])

    sharpe = float((log_ret.mean() / (log_ret.std() + 1e-12)) * math.sqrt(252)) if not log_ret.empty else float("nan")
    downside = log_ret[log_ret < 0]
    sortino = float((log_ret.mean() / (downside.std() + 1e-12)) * math.sqrt(252)) if len(downside) > 5 else float("nan")

    mdd = _max_drawdown(df["Close"])
    calmar = float(ann_ret / abs(mdd)) if (not math.isnan(ann_ret) and mdd < 0) else float("nan")

    var_95, cvar_95 = _var_cvar(log_ret, 0.05)
    skew = float(log_ret.skew()) if not log_ret.empty else float("nan")
    kurt = float(log_ret.kurt()) if not log_ret.empty else float("nan")

    return {
        "CAGR": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": mdd,
        "Calmar": calmar,
        "VaR_95_logret": var_95,
        "CVaR_95_logret": cvar_95,
        "Skew_logret": skew,
        "Kurt_logret": kurt,
    }


def aligned_returns(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for t, df in frames.items():
        tmp = df[["Date", "log_return"]].copy().rename(columns={"log_return": t})
        pieces.append(tmp)
    merged = pieces[0]
    for p in pieces[1:]:
        merged = merged.merge(p, on="Date", how="inner")
    return merged.dropna().reset_index(drop=True)


# ============================================================
# Forecast models
# ============================================================
def forecast_naive(train_close: np.ndarray, horizon: int) -> np.ndarray:
    last = float(train_close[-1])
    return np.array([last] * horizon, dtype=float)


def forecast_ma7(train_close: np.ndarray, horizon: int) -> np.ndarray:
    w = min(7, len(train_close))
    level = float(np.mean(train_close[-w:]))
    return np.array([level] * horizon, dtype=float)


def forecast_ewma_returns(train_close: np.ndarray, horizon: int, alpha: float = 0.2) -> np.ndarray:
    close = train_close.astype(float)
    lr = np.diff(np.log(close))
    if lr.size == 0:
        return forecast_naive(train_close, horizon)
    ewma = lr[0]
    for x in lr[1:]:
        ewma = alpha * x + (1 - alpha) * ewma
    out = []
    price = float(close[-1])
    for _ in range(horizon):
        price *= math.exp(float(ewma))
        out.append(price)
    return np.array(out, dtype=float)


def forecast_gbm(
    train_close: np.ndarray,
    horizon: int,
    seed: int,
    n_sims: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = train_close.astype(float)
    if len(close) < 30:
        mean = forecast_ma7(close, horizon)
        return mean, mean, mean

    lr = np.diff(np.log(close))
    mu = float(np.mean(lr))
    sigma = float(np.std(lr) + 1e-12)

    last = float(close[-1])
    sims = np.zeros((n_sims, horizon), dtype=float)

    for s in range(n_sims):
        price = last
        for d in range(horizon):
            shock = rng.normal(mu, sigma)
            price *= math.exp(float(shock))
            sims[s, d] = price

    mean = sims.mean(axis=0)
    lo = np.quantile(sims, 0.10, axis=0)
    hi = np.quantile(sims, 0.90, axis=0)
    return mean, lo, hi


# ============================================================
# Walk-forward backtest (multi-model)
# ============================================================
def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape}


def walk_forward_backtest_models(
    df: pd.DataFrame,
    models: Tuple[str, ...],
    horizon: int,
    steps: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    data = df.dropna(subset=["Close"]).copy()
    close = data["Close"].to_numpy(dtype=float)

    min_train = 90
    total_needed = min_train + steps * horizon + horizon
    if len(close) < total_needed:
        steps = max(1, (len(close) - min_train - horizon) // horizon)

    y_true_all: List[float] = []
    preds_all: Dict[str, List[float]] = {m: [] for m in models}

    for i in range(steps):
        cutoff = min_train + i * horizon
        train = close[:cutoff]
        test = close[cutoff: cutoff + horizon]
        if len(test) < horizon:
            break

        y_true_all.extend(test.tolist())

        for m in models:
            if m == "naive":
                pred = forecast_naive(train, horizon)
            elif m == "ma7":
                pred = forecast_ma7(train, horizon)
            elif m == "ewma_ret":
                pred = forecast_ewma_returns(train, horizon, alpha=0.2)
            elif m == "gbm":
                pred, _, _ = forecast_gbm(train, horizon, seed=seed + i, n_sims=1500)
            else:
                continue
            preds_all[m].extend(pred.tolist())

    y_true = np.array(y_true_all, dtype=float)
    out: Dict[str, Dict[str, float]] = {}
    for m, plist in preds_all.items():
        if len(plist) != len(y_true):
            continue
        out[m] = _error_metrics(y_true, np.array(plist, dtype=float))
    return out


# ============================================================
# Plotting
# ============================================================
def _clip_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return df
    last = pd.to_datetime(df["Date"].max())
    start = last - pd.Timedelta(days=days)
    return df[pd.to_datetime(df["Date"]) >= start].copy()


def save_plotly(fig: go.Figure, path: Path, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info(f"Saved plot: {path}")


def plot_ticker_dashboard(df: pd.DataFrame, ticker: str) -> go.Figure:
    use = df.dropna(subset=["Close"]).copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Close + MA(7/20)",
            "Rolling Volatility (20d, ann.)",
            "Drawdown",
            "Log-Return Distribution",
        )
    )

    fig.add_trace(go.Scatter(x=use["Date"], y=use["Close"], mode="lines", name="Close"), row=1, col=1)
    if "ma_7" in use.columns:
        fig.add_trace(go.Scatter(x=use["Date"], y=use["ma_7"], mode="lines", name="MA7"), row=1, col=1)
    if "ma_20" in use.columns:
        fig.add_trace(go.Scatter(x=use["Date"], y=use["ma_20"], mode="lines", name="MA20"), row=1, col=1)

    if "vol_20" in use.columns:
        fig.add_trace(go.Scatter(x=use["Date"], y=use["vol_20"], mode="lines", name="Vol20"), row=1, col=2)

    if "drawdown" in use.columns:
        fig.add_trace(go.Scatter(x=use["Date"], y=use["drawdown"], mode="lines", name="Drawdown"), row=2, col=1)

    lr = use["log_return"].dropna()
    if not lr.empty:
        hist = np.histogram(lr.to_numpy(), bins=50)
        mids = (hist[1][:-1] + hist[1][1:]) / 2
        fig.add_trace(go.Bar(x=mids, y=hist[0], name="LogRet Hist"), row=2, col=2)

    fig.update_layout(
        title=f"{ticker} — Dashboard",
        template="plotly_white",
        height=780,
        legend_orientation="h",
    )
    return fig


def plot_ohlc(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=go.Ohlc(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker
        )
    )
    fig.update_layout(
        title=f"{ticker} — OHLC (Daily)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=520,
    )
    return fig


def plot_intraday(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = _flatten_yfinance_columns(df.copy())
    if "DateTime" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{ticker}: intraday missing DateTime/Close.")
    fig = px.line(df, x="DateTime", y="Close", title=f"{ticker} — Intraday Close")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def plot_normalized_performance(frames: Dict[str, pd.DataFrame]) -> go.Figure:
    perf = None
    for t, df in frames.items():
        tmp = df[["Date", "Close"]].dropna().sort_values("Date").copy()
        base = float(tmp["Close"].iloc[0])
        tmp[t] = (tmp["Close"] / base) * 100.0
        tmp = tmp[["Date", t]]
        perf = tmp if perf is None else perf.merge(tmp, on="Date", how="inner")

    fig = px.line(perf, x="Date", y=list(frames.keys()), title="Normalized Performance (Start = 100)")
    fig.update_layout(template="plotly_white", height=480)
    return fig


def plot_returns_correlation(ret_df: pd.DataFrame) -> go.Figure:
    corr = ret_df.drop(columns=["Date"]).corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation of Daily Log Returns")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def plot_rolling_correlation(ret_df: pd.DataFrame, a: str, b: str, window: int = 30) -> go.Figure:
    x = ret_df[["Date", a, b]].copy()
    x["roll_corr"] = x[a].rolling(window).corr(x[b])
    fig = px.line(x.dropna(), x="Date", y="roll_corr", title=f"Rolling Correlation ({window}d): {a} vs {b}")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def plot_forecast(df: pd.DataFrame, ticker: str, horizon: int, seed: int) -> go.Figure:
    data = df.dropna(subset=["Close"]).copy()
    close = data["Close"].to_numpy(dtype=float)
    mean, lo, hi = forecast_gbm(close, horizon, seed=seed, n_sims=2000)

    last_date = pd.to_datetime(data["Date"].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    hist = data.tail(140).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], mode="lines", name="Close (history)"))
    fig.add_trace(go.Scatter(x=future_dates, y=mean, mode="lines", name="GBM mean", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=hi, mode="lines", name="P90", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=future_dates, y=lo, mode="lines", name="P10", fill="tonexty", line=dict(width=0)))

    fig.update_layout(
        title=f"{ticker} — Forecast (GBM baseline) + 10–90% band",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=520,
        legend_orientation="h",
    )
    return fig


# ============================================================
# Reporting / Index page
# ============================================================
def _to_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def markdown_to_basic_html(md_text: str) -> str:
    # Minimal, dependency-free rendering (keeps it readable in-browser).
    # We escape HTML and wrap in <pre>. Not pretty, but always works.
    escaped = html.escape(md_text)
    return f"<html><head><meta charset='utf-8'><title>Summary</title></head><body><pre>{escaped}</pre></body></html>"


def write_index_html(
    plots: List[Tuple[str, Path]],
    reports: List[Tuple[str, Path]],
    outdir: Path,
    outpath: Path,
    logger: logging.Logger,
) -> None:
    # Validate targets exist (prevents dead links)
    missing = []
    for _, p in plots + reports:
        if not (outdir / p).exists():
            missing.append(str(p))
    if missing:
        logger.warning("Index contains missing targets:\n" + "\n".join(missing))

    lines = [
        "<html><head><meta charset='utf-8'><title>Market Analysis Artifacts</title></head><body>",
        "<h1>Market Analysis Artifacts</h1>",
        "<h2>Plots</h2><ul>",
    ]
    for name, p in plots:
        rel = p.as_posix()
        lines.append(f"<li><a href='{rel}' target='_blank'>{html.escape(name)}</a></li>")
    lines.append("</ul><h2>Reports</h2><ul>")
    for name, p in reports:
        rel = p.as_posix()
        lines.append(f"<li><a href='{rel}' target='_blank'>{html.escape(name)}</a></li>")
    lines.append("</ul></body></html>")
    outpath.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(cfg: AnalysisConfig, logger: logging.Logger) -> Path:
    """
    Runs the full pipeline and returns the path to the main index.html.
    """
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tickers: {cfg.tickers}")
    logger.info(f"Range: {cfg.start} → {cfg.end} (end defaults to today unless --end provided)")
    logger.info(f"Artifacts: {cfg.outdir.resolve()}")

    data_dir = cfg.outdir / "data"
    reports_dir = cfg.outdir / "reports"
    plots_dir = cfg.outdir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    perf_rows = []
    backtest_rows = []

    for t in cfg.tickers:
        raw = download_history(
            ticker=t,
            start=cfg.start,
            end=cfg.end,
            logger=logger,
            cache_dir=cfg.cache_dir,
            cache_ttl_hours=cfg.cache_ttl_hours,
            refresh_cache=cfg.refresh_cache,
        )
        feat = add_features(raw)
        frames[t] = feat

        # Save daily artifacts
        try:
            feat.to_parquet(data_dir / f"{safe_name(t)}_daily.parquet", index=False)
        except Exception:
            pass
        feat.to_csv(data_dir / f"{safe_name(t)}_daily.csv", index=False)

        stats = compute_performance_stats(feat)
        perf_rows.append({"Ticker": t, **stats})

        bt = walk_forward_backtest_models(
            feat,
            models=cfg.models,
            horizon=cfg.backtest_horizon,
            steps=cfg.backtest_steps,
            seed=cfg.seed,
        )
        for model_name, m in bt.items():
            backtest_rows.append({"Ticker": t, "Model": model_name, **m})

    perf_df = pd.DataFrame(perf_rows)
    bt_df = pd.DataFrame(backtest_rows).sort_values(["Ticker", "RMSE"])

    perf_df.to_csv(reports_dir / "performance_risk_summary.csv", index=False)
    bt_df.to_csv(reports_dir / "backtest_model_comparison.csv", index=False)

    # Metrics JSON (robust serialization)
    metrics = {
        "config": to_jsonable(cfg.__dict__),
        "performance_risk_summary": to_jsonable(perf_rows),
        "backtest_model_comparison": to_jsonable(backtest_rows),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Markdown + HTML summary
    md_lines = []
    md_lines.append("# Market Analysis Report\n")
    md_lines.append(f"Date range: **{cfg.start} → {cfg.end}**\n")
    md_lines.append("## Performance & Risk Summary\n")
    md_lines.append(_to_markdown_table(perf_df.round(6)))
    md_lines.append("\n## Walk-forward Backtest (Lower is better)\n")
    md_lines.append(_to_markdown_table(bt_df.round(6)))
    summary_md = "\n".join(md_lines)
    (reports_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    (reports_dir / "summary.html").write_text(markdown_to_basic_html(summary_md), encoding="utf-8")

    # Plots
    plots_written: List[Tuple[str, Path]] = []
    reports_written: List[Tuple[str, Path]] = [
        ("Summary (HTML)", Path("reports/summary.html")),
        ("Summary (Markdown)", Path("reports/summary.md")),
        ("Performance & Risk (CSV)", Path("reports/performance_risk_summary.csv")),
        ("Backtest Comparison (CSV)", Path("reports/backtest_model_comparison.csv")),
        ("Metrics (JSON)", Path("reports/metrics.json")),
    ]

    for t, df in frames.items():
        t_safe = safe_name(t)
        clipped = _clip_last_days(df, cfg.plot_last_days)

        if all(c in clipped.columns for c in ["Open", "High", "Low", "Close"]):
            p = Path(f"plots/{t_safe}_ohlc.html")
            save_plotly(plot_ohlc(clipped, t), cfg.outdir / p, logger)
            plots_written.append((f"{t} OHLC", p))

        p = Path(f"plots/{t_safe}_dashboard.html")
        save_plotly(plot_ticker_dashboard(clipped, t), cfg.outdir / p, logger)
        plots_written.append((f"{t} Dashboard", p))

        p = Path(f"plots/{t_safe}_forecast_gbm.html")
        save_plotly(plot_forecast(df, t, cfg.forecast_days, cfg.seed), cfg.outdir / p, logger)
        plots_written.append((f"{t} Forecast (GBM baseline)", p))

        intraday = download_intraday(
            ticker=t,
            interval=cfg.intraday_interval,
            logger=logger,
            cache_dir=cfg.cache_dir,
            ttl_minutes=cfg.intraday_cache_ttl_minutes,
            refresh_cache=cfg.refresh_cache,
        )
        if not intraday.empty and ("DateTime" in intraday.columns) and ("Close" in intraday.columns):
            try:
                p = Path(f"plots/{t_safe}_intraday.html")
                save_plotly(plot_intraday(intraday, t), cfg.outdir / p, logger)
                plots_written.append((f"{t} Intraday", p))
            except Exception as e:
                logger.warning(f"{t}: intraday plot skipped: {e}")
        else:
            logger.warning(f"{t}: intraday unavailable (closed/delayed/rate limited).")

    if len(frames) >= 2:
        ret = aligned_returns(frames)

        p = Path("plots/normalized_performance.html")
        save_plotly(plot_normalized_performance(frames), cfg.outdir / p, logger)
        plots_written.append(("Normalized Performance", p))

        p = Path("plots/returns_correlation_heatmap.html")
        save_plotly(plot_returns_correlation(ret), cfg.outdir / p, logger)
        plots_written.append(("Returns Correlation Heatmap", p))

        keys = list(frames.keys())
        a, b = keys[0], keys[1]
        p = Path(f"plots/rolling_correlation_30d_{safe_name(a)}_vs_{safe_name(b)}.html")
        save_plotly(plot_rolling_correlation(ret, a, b, window=30), cfg.outdir / p, logger)
        plots_written.append((f"Rolling Corr 30d ({a} vs {b})", p))

    # IMPORTANT FIX: put index.html at artifacts/index.html (NOT inside artifacts/plots/)
    index_path = cfg.outdir / "index.html"
    write_index_html(
        plots=plots_written,
        reports=reports_written,
        outdir=cfg.outdir,
        outpath=index_path,
        logger=logger,
    )
    logger.info(f"Saved index: {index_path}")

    if cfg.open_index:
        try:
            webbrowser.open(index_path.resolve().as_uri())
        except Exception:
            pass

    logger.info("Done.")
    return index_path


# ============================================================
# API (FastAPI) — optional
# ============================================================
def serve_api(cfg: AnalysisConfig, logger: logging.Logger, host: str, port: int) -> None:
    """
    Starts a local API that serves artifacts + allows rerunning the pipeline.

    Install once:
      pip install fastapi uvicorn

    Run:
      python main.py --serve-api --open-index
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        import uvicorn
    except Exception as e:
        raise RuntimeError(
            "FastAPI/uvicorn not installed. Run:\n"
            "  python -m pip install fastapi uvicorn\n"
            f"Original error: {e}"
        )

    app = FastAPI(title="Market Analysis API", version="1.0")
    lock = threading.Lock()

    # Serve all artifacts at /artifacts/*
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=str(cfg.outdir), html=True), name="artifacts")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        # redirect-like simple HTML
        return HTMLResponse(
            "<html><body>"
            "<h2>Market Analysis API</h2>"
            "<ul>"
            "<li><a href='/index' target='_blank'>Open dashboard index</a></li>"
            "<li><a href='/metrics' target='_blank'>Latest metrics.json</a></li>"
            "<li><a href='/artifacts' target='_blank'>Browse artifacts folder</a></li>"
            "</ul>"
            "</body></html>"
        )

    @app.get("/index")
    def index():
        index_path = cfg.outdir / "index.html"
        if not index_path.exists():
            return HTMLResponse(
                "<html><body><p>index.html not found yet. Run POST /run first.</p></body></html>",
                status_code=404,
            )
        return FileResponse(str(index_path), media_type="text/html")

    @app.get("/metrics")
    def metrics():
        p = cfg.outdir / "reports" / "metrics.json"
        if not p.exists():
            return JSONResponse({"error": "metrics.json not found yet. Run POST /run first."}, status_code=404)
        return FileResponse(str(p), media_type="application/json")

    @app.post("/run")
    def run(refresh_cache: bool = False):
        # Synchronous run with a lock (prevents overlapping executions)
        with lock:
            run_cfg = dataclasses.replace(cfg, refresh_cache=bool(refresh_cache), open_index=False)
            index_path = run_pipeline(run_cfg, logger)
            return {
                "status": "completed",
                "index": "/index",
                "artifacts_index_file": str(index_path),
                "metrics": "/metrics",
            }

    logger.info(f"Starting API on http://{host}:{port}")
    logger.info("Open: /index  |  Run pipeline: POST /run  |  Metrics: /metrics")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ============================================================
# CLI
# ============================================================
def parse_args() -> Tuple[AnalysisConfig, str, bool, str, int]:
    p = argparse.ArgumentParser(description="Advanced market analytics pipeline with risk + backtest + dashboards + API.")

    p.add_argument("--tickers", nargs="+", default=["BTC-USD", "AAPL"])
    p.add_argument("--days", type=int, default=365, help="Lookback window in days.")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today).")

    p.add_argument("--intraday-interval", type=str, default="5m")
    p.add_argument("--cache-ttl-hours", type=int, default=12)
    p.add_argument("--intraday-cache-ttl-minutes", type=int, default=45)
    p.add_argument("--refresh-cache", action="store_true", help="Ignore caches and re-download.")

    p.add_argument("--outdir", type=str, default="artifacts")
    p.add_argument("--cache-dir", type=str, default=".cache_market")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forecast-days", type=int, default=15)
    p.add_argument("--backtest-horizon", type=int, default=15)
    p.add_argument("--backtest-steps", type=int, default=8)
    p.add_argument("--models", nargs="+", default=["naive", "ma7", "ewma_ret", "gbm"])

    p.add_argument("--plot-last-days", type=int, default=180)
    p.add_argument("--open-index", action="store_true", help="Open artifacts/index.html after run.")
    p.add_argument("--log-level", type=str, default="INFO")

    # API options
    p.add_argument("--serve-api", action="store_true", help="Start FastAPI server to serve artifacts and run pipeline.")
    p.add_argument("--api-host", type=str, default="127.0.0.1")
    p.add_argument("--api-port", type=int, default=8000)

    args = p.parse_args()

    # End date defaults to current date at runtime
    end_dt = date.today() if args.end is None else datetime.strptime(args.end, "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=int(args.days))

    cfg = AnalysisConfig(
        tickers=tuple(args.tickers),
        start=start_dt,
        end=end_dt,
        intraday_interval=args.intraday_interval,
        cache_ttl_hours=int(args.cache_ttl_hours),
        intraday_cache_ttl_minutes=int(args.intraday_cache_ttl_minutes),
        refresh_cache=bool(args.refresh_cache),
        outdir=Path(args.outdir),
        cache_dir=Path(args.cache_dir),
        seed=int(args.seed),
        forecast_days=int(args.forecast_days),
        backtest_horizon=int(args.backtest_horizon),
        backtest_steps=int(args.backtest_steps),
        models=tuple(args.models),
        plot_last_days=int(args.plot_last_days),
        open_index=bool(args.open_index),
    )
    return cfg, args.log_level, bool(args.serve_api), args.api_host, int(args.api_port)


if __name__ == "__main__":
    cfg, level, serve, host, port = parse_args()
    logger = setup_logger(level)

    if serve:
        # API mode
        serve_api(cfg, logger, host=host, port=port)
    else:
        # Script mode
        run_pipeline(cfg, logger)
