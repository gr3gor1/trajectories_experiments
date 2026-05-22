import argparse
import pandas as pd
from datetime import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a CSV file to rows whose 't' timestamp falls within a given time range."
    )
    parser.add_argument("--input", default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/ASV_logs.csv", help="Path to the input CSV file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for the output CSV. Defaults to <input>_filtered.csv."
    )
    parser.add_argument(
        "--start",
        default="11:54:06",
        metavar="HH:MM:SS",
        help="Start of the time range to keep (default: 11:54:06)."
    )
    parser.add_argument(
        "--end",
        default="11:56:09",
        metavar="HH:MM:SS",
        help="End of the time range to keep (default: 11:56:09)."
    )
    return parser.parse_args()

def parse_time(time_str: str) -> time:
    parts = time_str.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Invalid time format '{time_str}', expected HH:MM:SS.")
    return time(int(parts[0]), int(parts[1]), int(parts[2]))

def filter_by_time(input_csv: str, output_csv: str, start_time: time, end_time: time):
    df = pd.read_csv(input_csv)

    df["t"] = pd.to_datetime(df["t"], utc=True)

    t_only = df["t"].dt.time
    mask = (t_only >= start_time) & (t_only <= end_time)
    filtered = df[mask]

    filtered.to_csv(output_csv, index=False)
    print(f"Kept {len(filtered)} / {len(df)} rows.")
    print(f"Saved to: {output_csv}")

if __name__ == "__main__":
    args = parse_args()

    start_time = parse_time(args.start)
    end_time   = parse_time(args.end)

    output_path = args.output
    if output_path is None:
        base = args.input.rsplit(".", 1)[0]
        output_path = f"{base}_filtered.csv"

    filter_by_time(args.input, output_path, start_time, end_time)