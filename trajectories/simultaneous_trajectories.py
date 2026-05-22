"""
process_asv_logs.py

Reads an ASV log file (.xlsx or .csv), separates trajectories by MMSI,
detects missing timesteps (gap threshold: 2 seconds), and outputs only
the portions where both objects are moving simultaneously with no gaps.

The original data format is fully preserved, including:
- Integer columns stay as integers (no float conversion)
- The 'sog' column preserves mixed string/numeric values, including values
  that Excel may have misread as dates (e.g. '1.08' stored as Aug 1 date)
- Timestamp strings are written exactly as in the source
"""

import argparse
import datetime
import struct
import binascii
import pandas as pd
from pathlib import Path

INT_COLS = {'station_id', 'mmsi', 'cog', 'heading', 'status'}

def decode_wkb(hex_str: str):
    """
    Decode an EWKB hex geometry string and return (longitude, latitude).
    This is used as the ground truth for lon/lat, since the raw integer
    columns in the source file suffer from variable-digit truncation.
    """
    data = binascii.unhexlify(hex_str)
    fmt = '<' if data[0] == 1 else '>'
    wkb_type = struct.unpack_from(fmt + 'I', data, 1)[0]
    offset = 9 if bool(wkb_type & 0x20000000) else 5  # skip SRID if present
    return struct.unpack_from(fmt + 'dd', data, offset)


def _decode_sog_date(val: datetime.datetime, number_format: str) -> str:
    """
    Excel sometimes misreads decimal sog values (e.g. '1.08') as dates.
    This reverses that: datetime(2026,8,1) with format 'dd.mm' -> '01.08'
    """
    d, m = val.day, val.month
    return f'{d:02d}.{m:02d}' if 'dd' in number_format else f'{d}.{m}'


def read_xlsx_preserve(path: str) -> pd.DataFrame:
    """
    Read an xlsx file via openpyxl to preserve exact cell values.

    Handles two known quirks in ASV log files:
    1. The 'sog' column contains decimal strings that Excel may have
       auto-converted to date serial numbers (e.g. '1.08' -> date Aug 1).
       These are decoded back to their original string representation.
    2. Integer columns (mmsi, lon, lat, etc.) are returned as proper
       integers, not floats, even when some rows contain NaN.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]

    rows = []
    for ws_row in ws.iter_rows(min_row=2, values_only=False):
        row = {}
        for cell, col in zip(ws_row, headers):
            val = cell.value
            if col == 'sog' and isinstance(val, datetime.datetime):
                val = _decode_sog_date(val, cell.number_format)
            elif col in INT_COLS and val is not None:
                val = int(val)
            row[col] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=headers)

    # Use nullable Int64 so integer columns tolerate NaN without becoming float
    for col in INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    # Drop fully-empty trailing rows (artefacts of some xlsx exports)
    df = df.dropna(subset=['t', 'mmsi'])
    return df


def load_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in ('.xlsx', '.xlsm', '.xls'):
        return read_xlsx_preserve(path)
    elif p.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        for col in INT_COLS:
            if col in df.columns:
                df[col] = df[col].astype('Int64')
        return df
    else:
        raise ValueError(f'Unsupported file type: {p.suffix}')


def get_continuous_segments(grp: pd.DataFrame, gap: pd.Timedelta) -> list:
    """Split a single-MMSI DataFrame into continuous segments."""
    grp = grp.sort_values('_t').reset_index(drop=True)
    seg_id = (grp['_t'].diff() > gap).cumsum()
    segments = []
    for _, seg in grp.groupby(seg_id):
        segments.append((seg['_t'].min(), seg['_t'].max(), seg))
    return segments


def find_simultaneous_segments(segs_a: list, segs_b: list) -> list:
    """Return clipped DataFrames for every overlapping (A, B) segment pair."""
    parts = []
    for (a_start, a_end, a_df) in segs_a:
        for (b_start, b_end, b_df) in segs_b:
            overlap_start = max(a_start, b_start)
            overlap_end   = min(a_end,   b_end)
            if overlap_start < overlap_end:
                parts.append(a_df[(a_df['_t'] >= overlap_start) & (a_df['_t'] <= overlap_end)])
                parts.append(b_df[(b_df['_t'] >= overlap_start) & (b_df['_t'] <= overlap_end)])
    return parts


def process(input_path: str, output_path: str, gap_seconds: float = 2.0) -> pd.DataFrame:
    gap = pd.Timedelta(seconds=gap_seconds)

    # Load while preserving original formats
    df = load_file(input_path)

    # Replace raw lon/lat integers with precise float coordinates decoded from geom.
    # The integer columns in the source are scaled by 1e7 but suffer from variable-digit
    # truncation; the geom WKB column is the reliable ground truth.
    if 'geom' in df.columns:
        coords = df['geom'].apply(decode_wkb)
        df['lon'] = coords.apply(lambda c: c[0])
        df['lat'] = coords.apply(lambda c: c[1])

    # Parse timestamps into a working column (keep original 't' string untouched)
    df['_t'] = pd.to_datetime(df['t'], utc=True)

    # Validate
    mmsis = df['mmsi'].dropna().unique()
    if len(mmsis) < 2:
        raise ValueError(f'Expected at least 2 distinct MMSI values, found: {mmsis}')
    if len(mmsis) > 2:
        print(f'Warning: {len(mmsis)} MMSI values found. Processing all pairwise overlaps.')

    # Segment each MMSI
    segs_by_mmsi = {}
    for mmsi, grp in df.groupby('mmsi'):
        segs = get_continuous_segments(grp, gap)
        segs_by_mmsi[mmsi] = segs
        print(f'  MMSI {mmsi}: {len(segs)} continuous segment(s)')
        for i, (s, e, seg) in enumerate(segs):
            print(f'    Segment {i}: {s} → {e}  ({len(seg)} rows)')

    # Find simultaneous overlaps across all MMSI pairs
    mmsi_list = list(segs_by_mmsi.keys())
    all_parts = []
    for i in range(len(mmsi_list)):
        for j in range(i + 1, len(mmsi_list)):
            parts = find_simultaneous_segments(
                segs_by_mmsi[mmsi_list[i]],
                segs_by_mmsi[mmsi_list[j]],
            )
            all_parts.extend(parts)

    if not all_parts:
        print('No simultaneous overlap found between any pair of MMSIs.')
        return pd.DataFrame(columns=df.columns)

    result = (
        pd.concat(all_parts)
        .sort_values(['_t', 'mmsi'])
        .reset_index(drop=True)
        .drop(columns=['_t'])
    )

    result.to_csv(output_path, index=False, float_format='%.7f')

    print(f'\nOutput: {output_path}  ({len(result)} rows total)')
    summary = result.groupby('mmsi')['t'].agg(['min', 'max', 'count'])
    summary.columns = ['start', 'end', 'rows']
    print(summary.to_string())

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract simultaneous, gap-free ASV trajectory segments.'
    )
    parser.add_argument('--input_file', '-i',  # optional argument
                        default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/ASV_logs_filtered.csv",
                        help='Path to input .xlsx or .csv file')
    parser.add_argument(
        'output_file',
        nargs='?',
        default=None,
        help='Path for output CSV (default: <input_stem>_simultaneous.csv)',
    )
    parser.add_argument(
        '--gap-seconds',
        type=float,
        default=2.0,
        help='Gap threshold in seconds to split trajectories (default: 2)',
    )
    args = parser.parse_args()

    output = args.output_file or str(
        Path(args.input_file).with_name(
            Path(args.input_file).stem + '_simultaneous.csv'
        )
    )

    process(args.input_file, output, args.gap_seconds)


if __name__ == '__main__':
    main()
