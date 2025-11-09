# app.py
import os
import io
import re
import gzip
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# =========================
# Page setup
# =========================
st.set_page_config(page_title="PSX Breakout App", layout="wide")
st.title("📈 PSX Breakout App")
st.caption("Screen and visualize price breakouts from PSX daily Market Summary files (.lis/.csv and sometimes .Z/.gz).")

st.markdown(
    """
    <style>
      .ag-theme-alpine { font-size: 11px; }
      .ag-theme-alpine .ag-header-cell-label { font-size: 11px; }
      .ag-theme-alpine .ag-cell {
        line-height: 1.2 !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- paths (Cloud-safe) ----------
APP_DIR = Path(__file__).parent
MASTER_CSV_REPO = APP_DIR / "psx_master.csv"            # committed file (read-only in Cloud)
MASTER_CSV_RUNTIME = Path(os.getenv("MASTER_CSV", "/tmp/psx_master.csv"))  # writable copy
SECTOR_CSV = APP_DIR / "sector_map_psx.csv"
DEFAULT_DATA_DIR = ""
USE_MASTER_DEFAULT = MASTER_CSV_RUNTIME.exists() or MASTER_CSV_REPO.exists()

# =========================
# Sector map (CSV → fallback)
# =========================
SECTOR_MAP_FALLBACK: Dict[str, str] = {
    "801": "Automobile Assembler",
    "802": "Automobile Parts & Accessories",
    "803": "Cable & Electrical Goods",
    "804": "Cement",
    "805": "Chemical",
    "806": "Close-End Mutual Fund",
    "807": "Commercial Banks",
    "808": "Engineering",
    "809": "Fertilizer",
    "810": "Food & Personal Care Products",
    "811": "Glass & Ceramics",
    "812": "Insurance",
    "813": "Inv. Banks / Inv. Cos. / Securities Cos.",
    "814": "Jute",
    "815": "Leasing Companies",
    "816": "Leather & Tanneries",
    "818": "Miscellaneous",
    "819": "Modarabas",
    "820": "Oil & Gas Exploration Companies",
    "821": "Oil & Gas Marketing Companies",
    "822": "Paper, Board & Packaging",
    "823": "Pharmaceuticals",
    "824": "Power Generation & Distribution",
    "825": "Refinery",
    "826": "Sugar & Allied Industries",
    "827": "Synthetic & Rayon",
    "828": "Technology & Communication",
    "829": "Textile Composite",
    "830": "Textile Spinning",
    "831": "Textile Weaving",
    "832": "Tobacco",
    "833": "Transport",
    "834": "Vanaspati & Allied Industries",
    "835": "Woollen",
    "836": "Real Estate Investment Trust",
    "837": "Exchange Traded Funds",
    "838": "Property",
}
FUT_SECTOR_CODES = {"0040", "0041"}  # deliverable futures groups
MONTH_SUFFIX_RE = re.compile(r"-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|NOVB|NOVC)$", re.IGNORECASE)

# =========================
# Helpers: parsing and normalization
# =========================
def _sanitize_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out[~out["symbol"].str.contains(MONTH_SUFFIX_RE, na=False)]
    if "sector_code" in out.columns:
        out["sector_code"] = out["sector_code"].astype(str).str.zfill(4)
        out = out[~out["sector_code"].isin(FUT_SECTOR_CODES)]
    return out


def load_sector_map() -> Dict[str, str]:
    try:
        df = pd.read_csv(SECTOR_CSV, dtype=str)
        df.columns = df.columns.str.strip().str.lower()
        if {"sector_code", "sector_name"}.issubset(df.columns):
            m: Dict[str, str] = {}
            for code, name in zip(df["sector_code"], df["sector_name"]):
                c = str(code).strip().lstrip("0")
                n = str(name).strip()
                if c and n:
                    m[c] = n
            if m:
                return m
    except Exception:
        pass
    return SECTOR_MAP_FALLBACK


SECTOR_MAP = load_sector_map()


def _ensure_sector_name(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cols_lower = out.columns.str.lower()

    if "sector_name" not in cols_lower and "sector" in cols_lower:
        for c in out.columns:
            if c.lower() == "sector":
                out.rename(columns={c: "sector_name"}, inplace=True)
                break

    if "sector_name" in out.columns:
        if out["sector_name"].fillna("").eq("").all() and "sector_code" in out.columns:
            sc = out["sector_code"].astype(str).str.lstrip("0")
            out["sector_name"] = sc.map(SECTOR_MAP).fillna("")
    elif "sector_code" in out.columns:
        sc = out["sector_code"].astype(str).str.lstrip("0")
        out["sector_name"] = sc.map(SECTOR_MAP).fillna("")
    else:
        out["sector_name"] = out.get("sector_name", "")
    return out


def _read_text_safely(path: Path) -> str:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data).decode("latin-1", errors="ignore")
        except Exception:
            pass
    if data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    return zf.read(name).decode("latin-1", errors="ignore")
        except Exception:
            pass
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def _parse_lis_text(text: str) -> pd.DataFrame:
    # DDMMMYYYY|SYMB|SECT|Company|Open|High|Low|Close|Volume|LDCP|...
    lines = [ln for ln in text.splitlines() if ln.count("|") >= 9]
    if not lines:
        return pd.DataFrame()

    raw = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep="|", header=None, engine="c", dtype=str, na_filter=False, on_bad_lines="skip",
    )
    if raw.empty or raw.shape[1] < 10:
        return pd.DataFrame()

    raw = raw.iloc[:, :10].copy()
    raw.columns = ["date", "symbol", "sector_code", "company", "open", "high", "low", "close", "volume", "ldcp"]
    for c in raw.columns:
        raw[c] = (
            raw[c].astype(str)
            .str.replace("\x00", "", regex=False)
            .str.replace("\u00A0", " ", regex=False)
            .str.strip()
        )

    d = pd.to_datetime(raw["date"], format="%d%b%Y", errors="coerce")
    if d.isna().all():
        d = pd.to_datetime(raw["date"], errors="coerce", dayfirst=True)
    raw["date"] = d.dt.date

    for c in ["open", "high", "low", "close", "ldcp", "volume"]:
        raw[c] = pd.to_numeric(raw[c].str.replace(",", "", regex=False), errors="coerce")

    raw["symbol"] = raw["symbol"].str.upper()
    raw["sector_code"] = raw["sector_code"].astype(str).str.zfill(4)
    return raw.dropna(subset=["date", "symbol", "close"]).reset_index(drop=True)


def _parse_any_file(path: Path) -> pd.DataFrame:
    p = path.as_posix().lower()
    try:
        if p.endswith((".lis", ".z", ".gz", ".txt")):
            return _parse_lis_text(_read_text_safely(path))
        elif p.endswith(".csv"):
            df = pd.read_csv(path, engine="c")
            df.columns = df.columns.str.lower().str.strip()
            if not {"symbol", "close"}.issubset(df.columns):
                return pd.DataFrame()
            out = pd.DataFrame()
            out["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
            out["close"] = pd.to_numeric(df["close"], errors="coerce")
            out["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
            for c in ["open","high","low","volume","ldcp","sector_code","company","sector","sector_name"]:
                if c in df.columns:
                    out[c] = df[c]
            out = _ensure_sector_name(out)
            return out.dropna(subset=["date", "symbol", "close"])
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_directory(dirpath: str) -> pd.DataFrame:
    files = []
    for r, _, fs in os.walk(dirpath):
        for f in fs:
            if f.lower().endswith((".lis", ".csv", ".z", ".gz", ".txt")):
                files.append(Path(r) / f)
    if not files:
        return pd.DataFrame()

    frames = []
    for fp in sorted(files):
        df = _parse_any_file(fp)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    df_all = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    if not df_all.empty:
        df_all["symbol"] = df_all["symbol"].astype(str).str.strip().str.upper()
        df_all = df_all[~df_all["symbol"].str.contains(MONTH_SUFFIX_RE, na=False)]
        df_all = _sanitize_symbols(df_all)
        if "company" not in df_all.columns:
            df_all["company"] = ""
        df_all = _ensure_sector_name(df_all)

    return df_all


# =========================
# Indicators
# =========================
def _donchian_status(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    hi_prev = high.rolling(n, min_periods=n).max().shift(1)
    lo_prev = low.rolling(n, min_periods=n).min().shift(1)
    status = pd.Series(index=close.index, dtype="object")
    status[close > hi_prev] = "Breakout"
    status[close < lo_prev] = "Breakdown"
    return status.fillna("Within Range")


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr = _atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()

    for i in range(1, len(df)):
        upper[i] = min(upper_basic[i], upper[i - 1]) if df["close"][i - 1] > upper[i - 1] else upper_basic[i]
        lower[i] = max(lower_basic[i], lower[i - 1]) if df["close"][i - 1] < lower[i - 1] else lower_basic[i]

    st_line = pd.Series(index=df.index, dtype="float")
    for i in range(len(df)):
        if i == 0:
            st_line.iat[i] = upper.iat[i]
        else:
            if df["close"].iat[i] > upper.iat[i - 1]:
                st_line.iat[i] = lower.iat[i]
            elif df["close"].iat[i] < lower[i - 1]:
                st_line.iat[i] = upper.iat[i]
            else:
                st_line.iat[i] = st_line.iat[i - 1]
    return st_line


@st.cache_data(show_spinner=False)
def compute_all_indicators(df: pd.DataFrame, n_daily=20, n_weekly=10, n_monthly=6) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()

        for c in ["open", "high", "low"]:
            if c not in g.columns or g[c].isna().all():
                g[c] = g["close"]
            else:
                g[c] = g[c].fillna(g["close"])

        g["daily_status"] = _donchian_status(g["high"], g["low"], g["close"], n_daily)

        gw = (
            g.set_index("date")
            .resample("W-FRI")
            .agg({"high": "max", "low": "min", "close": "last"})
            .dropna(subset=["close"])
        )
        gw["weekly_status"] = _donchian_status(gw["high"], gw["low"], gw["close"], n_weekly)
        gw["week_start"] = gw.index.to_period("W-FRI").start_time
        g["week_start"] = g["date"].dt.to_period("W-FRI").dt.start_time
        g = g.merge(gw[["week_start", "weekly_status"]], on="week_start", how="left").drop(columns=["week_start"])

        gm = (
            g.set_index("date")
            .resample("M")
            .agg({"high": "max", "low": "min", "close": "last"})
            .dropna(subset=["close"])
        )
        gm["monthly_status"] = _donchian_status(gm["high"], gm["low"], gm["close"], n_monthly)
        gm["month_start"] = gm.index.to_period("M").start_time
        g["month_start"] = g["date"].dt.to_period("M").dt.start_time
        g = g.merge(gm[["month_start", "monthly_status"]], on="month_start", how="left").drop(columns=["month_start"])

        g["vol_ma20"] = g["volume"].rolling(20, min_periods=10).mean() if "volume" in g.columns else np.nan

        for n, col in [(9, "ema9"), (21, "ema21"), (44, "ema44"), (100, "ema100"), (200, "ema200")]:
            g[col] = _ema(g["close"], n)

        g["supertrend_10_3"] = supertrend(g[["high", "low", "close"]], 10, 3.0)
        out.append(g)

    return pd.concat(out, ignore_index=True).sort_values(["symbol", "date"])


# =========================
# Sidebar – load data
# =========================
st.sidebar.header("Data")
data_dir = st.sidebar.text_input("Directory with daily files (.lis/.csv/.Z/.gz)", value=DEFAULT_DATA_DIR)
refresh_master = st.sidebar.button("Refresh master from directory", use_container_width=True)
use_master = st.sidebar.checkbox("Use master dataset if available", value=USE_MASTER_DEFAULT)

if data_dir and os.path.isdir(data_dir) and refresh_master:
    df_tmp = load_directory(data_dir)
    if df_tmp.empty:
        st.error("No rows parsed from this directory. Are there .lis/.csv/.Z/.gz files?")
    else:
        df_tmp = _sanitize_symbols(df_tmp)
        # start from whichever master exists (runtime preferred)
        if MASTER_CSV_RUNTIME.exists():
            old = pd.read_csv(MASTER_CSV_RUNTIME, parse_dates=["date"])
        elif MASTER_CSV_REPO.exists():
            old = pd.read_csv(MASTER_CSV_REPO, parse_dates=["date"])
        else:
            old = pd.DataFrame(columns=["symbol", "date"])

        if not old.empty:
            old["date"] = pd.to_datetime(old["date"]).dt.date
            old = _sanitize_symbols(old)
            df_tmp = (
                pd.concat([old, df_tmp], ignore_index=True)
                .drop_duplicates(subset=["symbol", "date"])
                .sort_values(["symbol", "date"])
            )
        MASTER_CSV_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        df_tmp.to_csv(MASTER_CSV_RUNTIME, index=False)
        st.success(f"Master dataset refreshed (saved to {MASTER_CSV_RUNTIME}). Rows: {len(df_tmp):,}")
        st.cache_data.clear()

# Load dataset
if use_master and (MASTER_CSV_RUNTIME.exists() or MASTER_CSV_REPO.exists()):
    p = MASTER_CSV_RUNTIME if MASTER_CSV_RUNTIME.exists() else MASTER_CSV_REPO
    df_all = pd.read_csv(p, parse_dates=["date"])
    df_all["date"] = df_all["date"].dt.date
    df_all = _sanitize_symbols(df_all)
    df_all = _ensure_sector_name(df_all)
elif data_dir and os.path.isdir(data_dir):
    df_all = load_directory(data_dir)
else:
    df_all = pd.DataFrame()

if df_all.empty:
    st.info("Provide a folder with .lis/.csv/.Z/.gz files, or use the master dataset.")
    st.stop()

# =========================
# Parameters
# =========================
st.sidebar.header("Parameters")
min_date = df_all["date"].min()
max_date = df_all["date"].max()

date_val = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
start_date, end_date = (date_val if isinstance(date_val, tuple) else (date_val, date_val))

donchian_n = int(st.sidebar.number_input("Donchian lookback (days)", 10, 200, 20, 1))
vol_mult = float(st.sidebar.slider("Min volume vs 20D MA", 0.0, 3.0, 1.0, 0.1))
chart_lookback_m = int(st.sidebar.slider("Chart lookback (months)", 1, 24, 2, 1))

mask_date = (df_all["date"] >= start_date) & (df_all["date"] <= end_date)
df_all = df_all.loc[mask_date].copy()
df_all = compute_all_indicators(df_all, n_daily=donchian_n, n_weekly=10, n_monthly=6)

latest_ts = pd.to_datetime(df_all["date"]).max()
latest_date = latest_ts.date()
latest = df_all[df_all["date"] == latest_ts].copy()

if vol_mult > 0 and {"volume", "vol_ma20"}.issubset(latest.columns):
    latest = latest[latest["volume"] >= vol_mult * latest["vol_ma20"].fillna(0)]

# =========================
# KPI + filters
# =========================
def _count_status(df: pd.DataFrame, col: str) -> Tuple[int, int]:
    up = int((df[col] == "Breakout").sum())
    dn = int((df[col] == "Breakdown").sum())
    return up, dn

d_up, d_dn = _count_status(latest, "daily_status")
w_up, w_dn = _count_status(latest, "weekly_status")
m_up, m_dn = _count_status(latest, "monthly_status")

kpi_cols = st.columns(6)
kpis = [
    ("Daily Breakouts", d_up, "btn_db", "Breakout", "daily_sel"),
    ("Daily Breakdowns", d_dn, "btn_dd", "Breakdown", "daily_sel"),
    ("Weekly Breakouts", w_up, "btn_wb", "Breakout", "weekly_sel"),
    ("Weekly Breakdowns", w_dn, "btn_wd", "Breakdown", "weekly_sel"),
    ("Monthly Breakouts", m_up, "btn_mb", "Breakout", "monthly_sel"),
    ("Monthly Breakdowns", m_dn, "btn_md", "Breakdown", "monthly_sel"),
]
for col, (label, count, key, sel_value, sel_state_key) in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f"""
            <div style="
                background-color:#f8f9fa;
                border:1px solid #dee2e6;
                border-radius:12px;
                padding:14px 8px 8px;
                text-align:center;
                box-shadow:0 1px 3px rgba(0,0,0,0.08);
            ">
                <div style="font-size:2rem;font-weight:600;color:#1a1a1a;margin-bottom:8px;">
                    {count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(label, key=key, use_container_width=True):
            st.session_state[sel_state_key] = sel_value

st.subheader(f"Breakouts on {latest_date}")

fcols = st.columns(4)
sector_values = ["All Sectors"] + sorted([s for s in latest.get("sector_name", pd.Series([])).dropna().unique() if s])
sector_choice = fcols[0].selectbox("Sector", sector_values, index=0)
daily_choice = fcols[1].selectbox(
    "Daily Status", ["All", "Breakout", "Breakdown", "Within Range"],
    index=["All", "Breakout", "Breakdown", "Within Range"].index(st.session_state.get("daily_sel", "Breakout")),
    key="daily_sel",
)
weekly_choice = fcols[2].selectbox(
    "Weekly Status", ["All", "Breakout", "Breakdown", "Within Range"],
    index=["All", "Breakout", "Breakdown", "Within Range"].index(st.session_state.get("weekly_sel", "All")),
    key="weekly_sel",
)
monthly_choice = fcols[3].selectbox(
    "Monthly Status", ["All", "Breakout", "Breakdown", "Within Range"],
    index=["All", "Breakout", "Breakdown", "Within Range"].index(st.session_state.get("monthly_sel", "All")),
    key="monthly_sel",
)
query = st.text_input("Search records (symbol/company contains):", value="").strip().upper()

filtered = latest.copy()
if sector_choice != "All Sectors" and "sector_name" in filtered.columns:
    filtered = filtered[filtered["sector_name"] == sector_choice]
if daily_choice != "All":
    filtered = filtered[filtered["daily_status"] == daily_choice]
if weekly_choice != "All":
    filtered = filtered[filtered["weekly_status"] == weekly_choice]
if monthly_choice != "All":
    filtered = filtered[filtered["monthly_status"] == monthly_choice]
if query:
    m = filtered["symbol"].str.contains(query, na=False) | filtered["company"].str.upper().str.contains(query, na=False)
    filtered = filtered[m]

# EMA flags (text used for styling/tooltip)
def pos_flag(close: pd.Series, ema: pd.Series, tol: float = 0.001) -> pd.Series:
    diff = (close - ema).abs() / ema.replace(0, np.nan)
    out = pd.Series("Above", index=close.index, dtype="object")
    out[close < ema] = "Below"
    out[diff <= tol] = "On"
    return out

for col, name in [("ema9", "EMA 9"), ("ema21", "EMA 21"), ("ema44", "EMA 44"), ("ema100", "EMA 100"), ("ema200", "EMA 200")]:
    filtered[name] = pos_flag(filtered["close"], filtered[col])

# =========================
# Two tables
# =========================
# 1) Trend summary
summary_cols = ["symbol", "sector_name", "close", "daily_status", "weekly_status", "monthly_status", "volume"]
summary_df = filtered[summary_cols].sort_values(["daily_status","weekly_status","monthly_status","symbol"]).reset_index(drop=True)
summary_df.rename(columns={"sector_name": "sector"}, inplace=True)

# 2) EMA analysis
ema_df = filtered[["symbol", "sector_name", "EMA 9", "EMA 21", "EMA 44", "EMA 100", "EMA 200"]].copy()
ema_df.rename(columns={"sector_name": "sector"}, inplace=True)

# helper status columns (hidden)
ema_df["ema9_status"]   = ema_df["EMA 9"]
ema_df["ema21_status"]  = ema_df["EMA 21"]
ema_df["ema44_status"]  = ema_df["EMA 44"]
ema_df["ema100_status"] = ema_df["EMA 100"]
ema_df["ema200_status"] = ema_df["EMA 200"]
for c in ["EMA 9","EMA 21","EMA 44","EMA 100","EMA 200"]:
    ema_df[c] = "●"
ema_df = ema_df.sort_values("symbol").reset_index(drop=True)

# ---- Trend summary grid (colored text via JsCode) ----
status_style_js = JsCode("""
function(params){
  const v = (params.value || '').toString();
  if (v === 'Breakout')  { return {'color': '#22c55e', 'fontWeight': '700'}; }
  if (v === 'Breakdown') { return {'color': '#ef4444', 'fontWeight': '700'}; }
  return {'color': '#111827', 'fontWeight': '600'};
}
""")

st.markdown("#### Trend summary")
gb1 = GridOptionsBuilder.from_dataframe(summary_df)
gb1.configure_default_column(filter=False, sortable=True, resizable=True, suppressMenu=True)
gb1.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
gb1.configure_column("symbol", width=110)
gb1.configure_column("sector", width=240)
gb1.configure_column("close", width=100, type=["numericColumn"])
gb1.configure_column("volume", width=120, type=["numericColumn"])
for c in ["daily_status", "weekly_status", "monthly_status"]:
    gb1.configure_column(c, header_name=c.replace("_", " ").title(), width=150, cellStyle=status_style_js)
gb1.configure_selection(selection_mode="single", use_checkbox=False)
gb1.configure_grid_options(domLayout="normal", rowHeight=30)
grid1 = AgGrid(
    summary_df,
    gridOptions=gb1.build(),
    theme="alpine",
    height=420,
    fit_columns_on_grid_load=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=False,
    key="grid_summary",
)

# ---- EMA grid (colored dots via JsCode) ----
def dot_style_js(status_field: str) -> JsCode:
    return JsCode(f"""
    function(params){{
      const v = params.data && params.data['{status_field}'] ? params.data['{status_field}'].toString() : '';
      if (v === 'Above') return {{'color':'#22c55e','fontWeight':'900','textAlign':'center'}};
      if (v === 'Below') return {{'color':'#ef4444','fontWeight':'900','textAlign':'center'}};
      if (v === 'On')    return {{'color':'#111827','fontWeight':'900','textAlign':'center'}};
      return {{'color':'#111827','fontWeight':'900','textAlign':'center'}};
    }}
    """)

st.markdown("#### EMA analysis")
gb2 = GridOptionsBuilder.from_dataframe(ema_df)
gb2.configure_default_column(filter=False, sortable=True, resizable=True, suppressMenu=True)
gb2.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
gb2.configure_column("symbol", width=110)
gb2.configure_column("sector", width=260)

def setup_ema_col(col: str, status_col: str):
    gb2.configure_column(
        col,
        header_name=col,
        width=90,                    # wider so header is fully visible
        tooltipField=status_col,
        cellStyle=dot_style_js(status_col),
    )
    gb2.configure_column(status_col, hide=True)

setup_ema_col("EMA 9",   "ema9_status")
setup_ema_col("EMA 21",  "ema21_status")
setup_ema_col("EMA 44",  "ema44_status")
setup_ema_col("EMA 100", "ema100_status")
setup_ema_col("EMA 200", "ema200_status")

gb2.configure_selection(selection_mode="single", use_checkbox=False)
gb2.configure_grid_options(domLayout="normal", rowHeight=30)
grid2 = AgGrid(
    ema_df,
    gridOptions=gb2.build(),
    theme="alpine",
    height=340,
    fit_columns_on_grid_load=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=False,
    key="grid_ema",
)

# =========================
# Selection handling
# =========================
def _pick_symbol(selected_rows) -> Optional[str]:
    if selected_rows is None:
        return None
    try:
        if isinstance(selected_rows, pd.DataFrame):
            return None if selected_rows.empty else selected_rows.iloc[0].get("symbol")
    except Exception:
        pass
    if isinstance(selected_rows, list):
        return None if len(selected_rows) == 0 else selected_rows[0].get("symbol")
    try:
        return selected_rows[0].get("symbol")
    except Exception:
        return None

selected_symbol = _pick_symbol(grid2.get("selected_rows"))
if selected_symbol is None:
    selected_symbol = _pick_symbol(grid1.get("selected_rows"))

# =========================
# Chart
# =========================
def plot_symbol(gdf: pd.DataFrame, title: str, donchian_n: int, lookback_months: int):
    gdf = gdf.sort_values("date").copy()
    hi_prev = gdf["high"].rolling(donchian_n, min_periods=donchian_n).max().shift(1)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=gdf["date"], open=gdf["open"], high=gdf["high"], low=gdf["low"], close=gdf["close"],
            name="Price", increasing_line_color="#2ca02c", decreasing_line_color="#d62728", showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gdf["date"], y=hi_prev, mode="lines",
            name=f"Donchian High (prev {donchian_n})", line=dict(color="#4F81BD", width=3),
        )
    )
    for col, name, color in [
        ("ema9", "EMA 9", "#16a34a"),
        ("ema21", "EMA 21", "#0ea5e9"),
        ("ema44", "EMA 44", "#ef4444"),
        ("ema100", "EMA 100", "#8b5cf6"),
        ("ema200", "EMA 200", "#f97316"),
    ]:
        fig.add_trace(go.Scatter(x=gdf["date"], y=gdf[col], mode="lines", name=name, line=dict(color=color, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=gdf["date"], y=gdf["supertrend_10_3"], mode="lines", name="SuperTrend (10,3)", line=dict(color="black", width=3)))

    # last N months visible by default + range buttons
    end_dt = pd.to_datetime(gdf["date"]).max()
    start_dt = end_dt - pd.DateOffset(months=lookback_months)
    start_dt = max(start_dt, pd.to_datetime(gdf["date"]).min())

    fig.update_layout(
        title=title, height=680, legend=dict(orientation="v"),
        margin=dict(l=40, r=20, t=60, b=40), xaxis_title="Date", yaxis_title="Price", uirevision=title,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=2, label="2m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
            rangeslider=dict(visible=False),
            type="date",
            range=[start_dt, end_dt],
        ),
    )
    return fig


if selected_symbol:
    series = df_all[df_all["symbol"] == selected_symbol].copy()
    st.plotly_chart(
        plot_symbol(series, f"{selected_symbol} — Donchian {donchian_n}", donchian_n, chart_lookback_m),
        use_container_width=True
    )
else:
    st.info("Select a row in either table to display the chart.")
