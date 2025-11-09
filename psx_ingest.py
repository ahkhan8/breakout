
import os
import io
import gzip
import argparse
import pandas as pd

SECTOR_MAP_PATH = os.path.join(os.path.dirname(__file__), "sector_map_psx.csv")
MASTER_CSV = os.path.join(os.path.dirname(__file__), "psx_master.csv")
COLS = ["date","symbol","sector_code","company","open","high","low","close","volume","ldcp"]

def read_text_safely(path: str) -> str:
    with open(path, "rb") as fh:
        head = fh.read(2)
        fh.seek(0)
        data = fh.read()
    if len(head) >= 2 and head[0] == 0x1F and head[1] == 0x8B:
        with gzip.open(path, "rt", errors="ignore") as gz:
            return gz.read().replace("\\x00", "").replace("\\u00A0", " ")
    for enc in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1"):
        try:
            return data.decode(enc).replace("\\x00", "").replace("\\u00A0", " ")
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore").replace("\\x00", "").replace("\\u00A0", " ")

def parse_lis(path: str) -> pd.DataFrame:
    text = read_text_safely(path)
    raw = pd.read_csv(io.StringIO(text),
                      sep="|",
                      header=None,
                      engine="python",
                      dtype=str,
                      na_filter=False,
                      on_bad_lines="skip")
    if raw.shape[1] < 10:
        for _ in range(10 - raw.shape[1]):
            raw[raw.shape[1]] = ""
    raw = raw.iloc[:, :10].copy()
    raw.columns = COLS

    for c in raw.columns:
        raw[c] = raw[c].astype(str).str.strip()

    raw["date"] = pd.to_datetime(raw["date"], format="%d%b%Y", errors="coerce")

    for c in ["open","high","low","close","volume","ldcp"]:
        raw[c] = pd.to_numeric(raw[c].str.replace(",","").str.replace("\\u00A0",""), errors="coerce")

    raw["symbol"] = raw["symbol"].str.upper()
    raw["sector_code"] = raw["sector_code"].str.zfill(4)

    try:
        sector = pd.read_csv(SECTOR_MAP_PATH, dtype=str)
        sector["sector_code"] = sector["sector_code"].str.zfill(4)
        raw = raw.merge(sector, on="sector_code", how="left")
    except Exception:
        raw["sector_name"] = None

    raw = raw.dropna(subset=["date","symbol","close"]).copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    return raw

def update_master(src_dir: str):
    src_dir = os.path.abspath(src_dir)
    all_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                 if f.lower().endswith((".lis",".z",".gz",".csv",".txt"))]
    frames = []
    for fp in sorted(all_files):
        try:
            if fp.lower().endswith((".lis",".z",".gz")):
                df = parse_lis(fp)
            else:
                df_raw = pd.read_csv(fp, engine="python")
                df_raw.columns = df_raw.columns.str.lower()
                if "symbol" in df_raw.columns and "close" in df_raw.columns:
                    df = pd.DataFrame({
                        "date": pd.to_datetime(df_raw.get("date"), errors="coerce").dt.date,
                        "symbol": df_raw["symbol"].astype(str).str.upper(),
                        "open": pd.to_numeric(df_raw.get("open"), errors="coerce"),
                        "high": pd.to_numeric(df_raw.get("high"), errors="coerce"),
                        "low":  pd.to_numeric(df_raw.get("low"), errors="coerce"),
                        "close":pd.to_numeric(df_raw.get("close"), errors="coerce"),
                        "volume":pd.to_numeric(df_raw.get("volume"), errors="coerce"),
                    })
                else:
                    df = pd.DataFrame()
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    if not frames:
        print("No rows parsed."); return

    new_data = pd.concat(frames, ignore_index=True)
    if os.path.exists(MASTER_CSV):
        master = pd.read_csv(MASTER_CSV, parse_dates=["date"])
        master["date"] = master["date"].dt.date
        combined = pd.concat([master, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = (combined
                .sort_values(["symbol","date"])
                .drop_duplicates(subset=["symbol","date"], keep="last")
                .reset_index(drop=True))
    combined.to_csv(MASTER_CSV, index=False)
    print("Updated master:", MASTER_CSV, "rows:", len(combined))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse PSX .lis/.Z/.gz and update master CSV")
    ap.add_argument("src_dir", help="Directory containing files")
    args = ap.parse_args()
    update_master(args.src_dir)
