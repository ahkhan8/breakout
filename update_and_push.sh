#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

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

