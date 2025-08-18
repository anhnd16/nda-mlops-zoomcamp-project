import os, time
import pandas as pd
from pathlib import Path

CAPTURE_PATH = os.getenv("CAPTURE_PATH", "data/capture/events.csv")
OUT_CURRENT = os.getenv("CUR_DATA", "data/current.csv")
WINDOW_SECS = int(os.getenv("MONITOR_WINDOW_SECS", 24*3600))  # last 24h by default


def build_current_window():
    now = int(time.time())
    if not Path(CAPTURE_PATH).exists():
        # Create an empty current file if no capture yet
        pd.DataFrame().to_csv(OUT_CURRENT, index=False)
        return OUT_CURRENT
    df = pd.read_csv(CAPTURE_PATH)
    if "ts" in df.columns:
        df = df[df["ts"] >= now - WINDOW_SECS]
    df.to_csv(OUT_CURRENT, index=False)
    return OUT_CURRENT


if __name__ == "__main__":
    print(build_current_window())