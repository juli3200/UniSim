# SCRIPT GENERATED WITH GPT-5

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

"""
graph.py

Create a performance graph with two lines (CPU and GPU) from a CSV file.

Usage:
    python graph.py path/to/data.csv --time-col time --cpu-col CPU --gpu-col GPU --out performance.png --show

The CSV should contain columns for CPU and GPU measurements and an optional timestamp/time column.
If no time column is provided, row index is used on the X axis.
"""


try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception as e:
    print("Missing dependency: please install pandas and matplotlib (pip install pandas matplotlib)")
    raise

def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_data(path: Path, time_col: str | None, cpu_col: str | None, gpu_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Auto-detect columns if not provided
    if time_col is None:
        time_col = find_column(df, ["time", "timestamp", "ts", "date", "t"])
    if cpu_col is None:
        cpu_col = find_column(df, ["CPU", "cpu", "Cpu", "cpu_percent", "cpu%"])
    if gpu_col is None:
        gpu_col = find_column(df, ["GPU", "gpu", "Gpu", "gpu_percent", "gpu%"])

    if cpu_col is None or gpu_col is None:
        raise ValueError("Could not find CPU and GPU columns automatically. Provide --cpu-col and --gpu-col.")

    # If time column exists, try to parse it to datetime, otherwise use index
    if time_col is not None and time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            # If parsing fails, keep as-is (could be numeric)
            pass
        df = df.set_index(time_col)
    else:
        # Ensure a simple numeric index for plotting
        df = df.reset_index(drop=True)

    # Convert cpu/gpu columns to numeric (coerce errors)
    df[cpu_col] = pd.to_numeric(df[cpu_col], errors="coerce")
    df[gpu_col] = pd.to_numeric(df[gpu_col], errors="coerce")

    # Keep only relevant columns
    return df[[cpu_col, gpu_col]].copy()

def plot_performance(df: pd.DataFrame, cpu_col: str, gpu_col: str, out_path: Path | None, show: bool):
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    x = df.index
    # Use actual column names as legend labels
    ax.plot(x, df[cpu_col], label=str(cpu_col), linewidth=1.8)
    ax.plot(x, df[gpu_col], label=str(gpu_col), linewidth=1.8)

    # X axis is Time, Y axis is Number of ligands
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of ligands")
    ax.set_title("Ligands over time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4)

    # If x is datetime-like, format the x-axis nicely
    if pd.api.types.is_datetime64_any_dtype(x) or hasattr(x, "tz") or (len(x) > 0 and isinstance(x[0], (pd.Timestamp,))):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S\n%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved graph to: {out_path}")
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot CPU and GPU performance from a CSV file.")
    parser.add_argument("csv", type=Path, help="Path to CSV file")
    parser.add_argument("--time-col", help="Name of time column (optional)")
    parser.add_argument("--cpu-col", help="Name of CPU column (optional)")
    parser.add_argument("--gpu-col", help="Name of GPU column (optional)")
    parser.add_argument("--out", type=Path, help="Output image file (png, svg). If omitted, saves as performance.png in same folder.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    try:
        df = load_data(args.csv, args.time_col, args.cpu_col, args.gpu_col)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(3)

    # Determine actual column names used (after load_data returned subset)
    cpu_col, gpu_col = df.columns.tolist()

    out_path = args.out or (args.csv.parent / "performance.png")
    try:
        plot_performance(df, cpu_col, gpu_col, out_path, args.show)
    except Exception as e:
        print(f"Error plotting data: {e}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()