git pull --rebase --autostash || true

DATA_DIR="data"
MASTER_CSV="psx_master.csv"

# --- Ensure sector_name backfilled before committing ---
python3 - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("psx_master.csv")
if not p.exists():
    raise SystemExit("❌ psx_master.csv not found, skipping sector backfill")

df = pd.read_csv(p, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"]).dt.date

if "sector_name" not in df.columns:
    df["sector_name"] = ""

# Build lookup from last known non-empty sector_name per symbol
tmp = df.loc[df["sector_name"].astype(str).str.strip() != "", ["symbol","date","sector_name"]].copy()
if not tmp.empty:
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values(["symbol","date"])
    last_known = tmp.groupby("symbol", as_index=True)["sector_name"].last()
    needs = df["sector_name"].astype(str).str.strip() == ""
    df.loc[needs, "sector_name"] = df.loc[needs, "symbol"].map(last_known).fillna(df.loc[needs, "sector_name"])

# Optional: fallback from sector_code
SECTOR_MAP = {
    "801":"Automobile Assembler","802":"Automobile Parts & Accessories","803":"Cable & Electrical Goods",
    "804":"Cement","805":"Chemical","806":"Close-End Mutual Fund","807":"Commercial Banks","808":"Engineering",
    "809":"Fertilizer","810":"Food & Personal Care Products","811":"Glass & Ceramics","812":"Insurance",
    "813":"Inv. Banks / Inv. Cos. / Securities Cos.","814":"Jute","815":"Leasing Companies",
    "816":"Leather & Tanneries","818":"Miscellaneous","819":"Modarabas","820":"Oil & Gas Exploration Companies",
    "821":"Oil & Gas Marketing Companies","822":"Paper, Board & Packaging","823":"Pharmaceuticals",
    "824":"Power Generation & Distribution","825":"Refinery","826":"Sugar & Allied Industries",
    "827":"Synthetic & Rayon","828":"Technology & Communication","829":"Textile Composite","830":"Textile Spinning",
    "831":"Textile Weaving","832":"Tobacco","833":"Transport","834":"Vanaspati & Allied Industries",
    "835":"Woollen","836":"Real Estate Investment Trust","837":"Exchange Traded Funds","838":"Property"
}
if "sector_code" in df.columns:
    sc = df["sector_code"].astype(str).str.strip().str.lstrip("0")
    df["sector_name"] = df["sector_name"].fillna(sc.map(SECTOR_MAP))

df.to_csv(p, index=False)
print("✅ Backfilled sector_name and saved to psx_master.csv")
PY

echo "📦 PSX Breakout — Update and Push"
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

git pull --rebase --autostash || true

DATA_DIR="data"
MASTER_CSV="psx_master.csv"

echo "📦 PSX Breakout — Update and Push"

python3 - <<'PY'
import os, warnings, pandas as pd
from psx_ingest import load_directory, _sanitize_symbols

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MASTER_CSV = "psx_master.csv"

if not os.path.isdir(DATA_DIR):
    raise SystemExit("❌ data/ folder not found.")

df_new = load_directory(DATA_DIR)
if df_new.empty:
    print("⚠️ No valid new data found.")
    raise SystemExit(0)

if os.path.exists(MASTER_CSV):
    df_old = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
    df_old = _sanitize_symbols(df_old)
    before = len(df_old)
    df = pd.concat([df_old, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=["symbol","date"]).sort_values(["symbol","date"])
    added = len(df) - before
    print(f"✅ Added {added:,} new rows.")
else:
    df = df_new.copy()
    print("📄 Creating new psx_master.csv")

df.to_csv(MASTER_CSV, index=False)
print(f"✅ Saved {len(df):,} rows total.")
PY

echo "🔁 Committing changes..."
git add psx_master.csv
git commit -m "Auto-update PSX data $(date +'%Y-%m-%d %H:%M')" || echo "No changes to commit."
git push

