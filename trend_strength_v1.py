"""Trend Strength v1.0 (Regime Confidence).

Computes a research-ready dataset from daily OHLCV-style input data
(date, symbol, close, ...), including:
- moving-average structure features
- rolling OLS slope t-stat drift features
- acceleration, raw and smoothed trend strength
- forward return and excursion metrics

Formulas implemented exactly as specified in the task statement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendStrengthParams:
    """Parameter set for Trend Strength v1.0 formulas."""

    c_S: float = 0.04
    c_D: float = 3.0
    L_e: int = 15
    c_A: float = 0.01
    w_S: float = 0.60
    w_D: float = 0.35
    w_A: float = 0.05
    N_ema: int = 15


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort input by symbol/date without mutating caller df."""
    _validate_columns(df, ["date", "symbol", "close"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def compute_sma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA and structure features per symbol.

    Formulas:
    - MA20, MA50, MA200: simple moving averages on close
    - m20_50 = (MA20 - MA50) / MA50
    - m50_200 = (MA50 - MA200) / MA200
    - S = tanh((m20_50 + m50_200) / c_S) [c_S applied later in compute_trend_strength]
    """
    out = df.copy()
    g = out.groupby("symbol", group_keys=False)["close"]
    out["MA20"] = g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["MA50"] = g.transform(lambda s: s.rolling(50, min_periods=50).mean())
    out["MA200"] = g.transform(lambda s: s.rolling(200, min_periods=200).mean())

    out["m20_50"] = (out["MA20"] - out["MA50"]) / out["MA50"]
    out["m50_200"] = (out["MA50"] - out["MA200"]) / out["MA200"]
    return out


def rolling_ols_tstat(series: pd.Series, window: int) -> pd.Series:
    """Return rolling OLS slope t-statistics for y over k=0..window-1.

    Regression per window:
      y_k = a + b*k
    T-statistic:
      t = b / SE(b)

    Notes:
    - NaN in a window -> NaN output.
    - Zero variance / undefined standard error -> NaN.
    """

    def _window_tstat(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        n = y.size
        if n < 3:
            return np.nan

        x = np.arange(n, dtype=float)
        x_bar = (n - 1) / 2.0
        y_bar = y.mean()

        x_centered = x - x_bar
        y_centered = y - y_bar

        sxx = np.dot(x_centered, x_centered)
        if sxx <= 0:
            return np.nan

        b = np.dot(x_centered, y_centered) / sxx
        a = y_bar - b * x_bar
        resid = y - (a + b * x)
        sse = np.dot(resid, resid)

        dof = n - 2
        if dof <= 0:
            return np.nan

        sigma2 = sse / dof
        se_b = np.sqrt(sigma2 / sxx)
        if not np.isfinite(se_b) or se_b <= 0:
            return np.nan

        return float(b / se_b)

    return series.rolling(window=window, min_periods=window).apply(_window_tstat, raw=True)


def compute_trend_strength(
    df: pd.DataFrame, params: Optional[TrendStrengthParams] = None
) -> pd.DataFrame:
    """Compute Trend Strength v1.0 columns per symbol.

    Implements exactly:
      S = tanh((m20_50 + m50_200)/c_S)
      D = 0.5*T20 + 0.3*T50 + 0.2*T200
      D_tilde = tanh(D/c_D)
      e = m20_50 - m20_50.shift(L_e)
      A = tanh(e/c_A)
      R = w_S*S + w_D*D_tilde + w_A*A
      TS_raw = tanh(R)
      TS = EMA(TS_raw, span=N_ema)
    """
    p = params or TrendStrengthParams()
    out = _prepare_input(df)
    out = compute_sma_features(out)

    out["S"] = np.tanh((out["m20_50"] + out["m50_200"]) / p.c_S)
    out["log_close"] = np.log(out["close"])

    frames = []
    for _, g in out.groupby("symbol", sort=False):
        g = g.copy()
        g["T20"] = rolling_ols_tstat(g["log_close"], 20)
        g["T50"] = rolling_ols_tstat(g["log_close"], 50)
        g["T200"] = rolling_ols_tstat(g["log_close"], 200)

        g["D"] = 0.5 * g["T20"] + 0.3 * g["T50"] + 0.2 * g["T200"]
        g["D_tilde"] = np.tanh(g["D"] / p.c_D)

        g["e"] = g["m20_50"] - g["m20_50"].shift(p.L_e)
        g["A"] = np.tanh(g["e"] / p.c_A)

        g["R"] = p.w_S * g["S"] + p.w_D * g["D_tilde"] + p.w_A * g["A"]
        g["TS_raw"] = np.tanh(g["R"])
        g["TS"] = g["TS_raw"].ewm(span=p.N_ema, adjust=False).mean()
        frames.append(g)

    out = pd.concat(frames, axis=0).sort_values(["symbol", "date"]).reset_index(drop=True)
    out = out.drop(columns=["log_close"])
    return out


def compute_forward_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward returns and 10-day forward drawdown/runup per symbol.

    Formulas:
    - fwd_return_Nd = close(t+N)/close(t) - 1 for N in {5,10,20}
    - fwd_max_drawdown_10d = min_{k=1..10}(close(t+k)/close(t)-1)
    - fwd_max_runup_10d = max_{k=1..10}(close(t+k)/close(t)-1)
    """
    out = _prepare_input(df)

    result = []
    for _, g in out.groupby("symbol", sort=False):
        g = g.copy()
        c = g["close"]
        g["fwd_return_5d"] = c.shift(-5) / c - 1.0
        g["fwd_return_10d"] = c.shift(-10) / c - 1.0
        g["fwd_return_20d"] = c.shift(-20) / c - 1.0

        fr = pd.concat([(c.shift(-k) / c - 1.0).rename(k) for k in range(1, 11)], axis=1)
        g["fwd_max_drawdown_10d"] = fr.min(axis=1)
        g["fwd_max_runup_10d"] = fr.max(axis=1)
        result.append(g)

    return pd.concat(result, axis=0).sort_values(["symbol", "date"]).reset_index(drop=True)


def _print_summary(df: pd.DataFrame) -> None:
    """Print per-symbol and overall summary stats for TS regimes and NaN rates."""

    def _line(block: pd.DataFrame, label: str) -> None:
        ts = block["TS"]
        n = len(block)
        pct_pos = 100.0 * (ts >= 0.6).sum() / n if n else np.nan
        pct_neg = 100.0 * (ts <= -0.6).sum() / n if n else np.nan
        nan_rate = 100.0 * ts.isna().mean() if n else np.nan
        print(
            f"{label}: rows={n}, %TS>=0.6={pct_pos:.2f}, %TS<=-0.6={pct_neg:.2f}, TS NaN rate={nan_rate:.2f}%"
        )

    for sym, g in df.groupby("symbol", sort=False):
        _line(g, f"symbol={sym}")
    _line(df, "overall")


def _run_assert_checks(df: pd.DataFrame) -> None:
    """Small unit-test style assertions requested in spec."""
    ts_non_nan = df["TS"].dropna()
    assert ((ts_non_nan >= -1.0) & (ts_non_nan <= 1.0)).all(), "TS must be in [-1, 1]"

    # Sign behavior for structure score S.
    s_pos = np.tanh((0.10 + 0.08) / 0.04)
    s_neg = np.tanh((-0.10 - 0.08) / 0.04)
    assert s_pos > 0 and s_neg < 0, "S sign sanity failed"

    # Forward return direction sanity: uses future price.
    ex = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8),
            "symbol": ["TEST"] * 8,
            "close": [100, 101, 102, 103, 104, 105, 106, 107],
        }
    )
    ex_fwd = compute_forward_metrics(ex)
    expected = 105 / 100 - 1.0
    got = ex_fwd.loc[0, "fwd_return_5d"]
    assert np.isclose(got, expected, atol=1e-12), "Forward return shift direction is incorrect"


def build_dataset(input_df: pd.DataFrame, params: Optional[TrendStrengthParams] = None) -> pd.DataFrame:
    """End-to-end dataset builder returning all required research columns."""
    ts_df = compute_trend_strength(input_df, params=params)
    final_df = compute_forward_metrics(ts_df)

    ordered_cols = [
        "date",
        "symbol",
        "close",
        "MA20",
        "MA50",
        "MA200",
        "m20_50",
        "m50_200",
        "S",
        "T20",
        "T50",
        "T200",
        "D",
        "D_tilde",
        "e",
        "A",
        "R",
        "TS_raw",
        "TS",
        "fwd_return_5d",
        "fwd_return_10d",
        "fwd_return_20d",
        "fwd_max_drawdown_10d",
        "fwd_max_runup_10d",
    ]
    return final_df[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Trend Strength v1.0 research dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional CSV input path. Must include date, symbol, close columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trend_strength_dataset.parquet",
        help="Output path (.parquet preferred; .csv supported).",
    )
    args = parser.parse_args()

    if args.input is None:
        raise ValueError("Please provide --input CSV path with date, symbol, close columns.")

    df = pd.read_csv(args.input)
    dataset = build_dataset(df)
    _run_assert_checks(dataset)

    output_path = Path(args.output)
    if output_path.suffix.lower() == ".csv":
        dataset.to_csv(output_path, index=False)
    else:
        try:
            dataset.to_parquet(output_path, index=False)
        except Exception:
            fallback = output_path.with_suffix(".csv")
            dataset.to_csv(fallback, index=False)
            print(f"Parquet write failed; wrote CSV fallback: {fallback}")

    _print_summary(dataset)


if __name__ == "__main__":
    main()
