import os, io, gzip, zipfile, re, pathlib
import pandas as pd

MONTH_SUFFIX_RE = re.compile(r"-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$", re.IGNORECASE)

def _sanitize_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out[~out["symbol"].str.contains(MONTH_SUFFIX_RE, na=False)]
    return out

def _read_text_safely(path: str) -> str:
    data = pathlib.Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":  # gzip
        return gzip.decompress(data).decode("latin-1", errors="ignore")
    if data[:2] == b"PK":        # zip
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                return zf.read(name).decode("latin-1", errors="ignore")
    return data.decode("latin-1", errors="ignore")

def _parse_lis_text(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.count("|") >= 9]
    if not lines:
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", header=None, engine="c", na_filter=False)
    raw = raw.iloc[:, :10]
    raw.columns = ["date","symbol","sector_code","company","open","high","low","close","volume","ldcp"]
    raw["date"] = pd.to_datetime(raw["date"], format="%d%b%Y", errors="coerce").dt.date
    for c in ["open","high","low","close","ldcp","volume"]:
        raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", ""), errors="coerce")
    return raw.dropna(subset=["date","symbol","close"])

def _parse_any_file(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith((".lis",".z",".gz",".txt")):
        return _parse_lis_text(_read_text_safely(path))
    elif p.endswith(".csv"):
        df = pd.read_csv(path)
        if "symbol" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df.dropna(subset=["date","symbol","close"])
    return pd.DataFrame()

def load_directory(dirpath: str) -> pd.DataFrame:
    files = []
    for root,_,fs in os.walk(dirpath):
        for f in fs:
            if f.lower().endswith((".lis",".csv",".z",".gz",".txt")):
                files.append(os.path.join(root,f))
    frames = []
    for fp in sorted(files):
        df = _parse_any_file(fp)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["symbol","date"]).sort_values(["symbol","date"])
    return _sanitize_symbols(df_all)