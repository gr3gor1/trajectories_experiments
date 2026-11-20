"""
Trajectory signals: distance between two vessels and how fast it is shrinking.

Two ways to get them:

* `ProximityTracker` — online. Feed it position fixes as they arrive
  (e.g. from a live AIS/GPS stream) and read the current distance and
  closure rate at any time.
* `TrajectoryLog`    — offline. Load a recorded log (CSV with t, mmsi, lon, lat
  or a WKB `geom` column), resampled to 1 Hz exactly like the evaluation
  scripts, and look up the reading for any timestamp or video frame.

Closure rate is d(distance)/dt in m/s: negative means the vessels are approaching.
"""

from __future__ import annotations

import binascii
import math
import struct
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class ProximityReading:
    distance_m: float
    closure_rate_ms: float  # < 0 → approaching


# ══════════════════════════════════════════════════════════════════════════════
#  Online
# ══════════════════════════════════════════════════════════════════════════════

class ProximityTracker:
    """
    Keeps the latest fix of each vessel and a short distance history.

        trk = ProximityTracker()
        trk.update("asv",  t, lon, lat)
        trk.update("boat", t, lon, lat)
        reading = trk.reading()        # None until both vessels have a fix

    The closure rate is the least-squares slope of distance over the last
    `window_s` seconds (robust to GPS jitter; with two 1-Hz samples it equals
    the plain difference used in the paper). Readings older than `max_age_s`
    are treated as unknown, so the router falls back to the safe level.
    """

    def __init__(self, window_s: float = 2.0, max_age_s: float = 5.0):
        self.window_s, self.max_age_s = window_s, max_age_s
        self.fixes: dict[object, tuple[float, float, float]] = {}  # id → (t, lon, lat)
        self.history: deque[tuple[float, float]] = deque()        # (t, distance)

    def update(self, vessel_id, t: float, lon: float, lat: float) -> None:
        """t in seconds (any monotonic clock shared by both vessels)."""
        self.fixes[vessel_id] = (t, lon, lat)
        if len(self.fixes) < 2:
            return
        (ta, loa, laa), (tb, lob, lab) = list(self.fixes.values())[-2:]
        now = max(ta, tb)
        self.history.append((now, haversine(loa, laa, lob, lab)))
        while self.history and self.history[0][0] < now - self.window_s:
            self.history.popleft()

    def reading(self, now: float | None = None) -> ProximityReading | None:
        if not self.history:
            return None
        t_last, d_last = self.history[-1]
        if now is not None and now - t_last > self.max_age_s:
            return None
        if len(self.history) < 2:
            return ProximityReading(d_last, 0.0)
        ts = np.array([t for t, _ in self.history])
        ds = np.array([d for _, d in self.history])
        if np.ptp(ts) == 0:
            return ProximityReading(d_last, 0.0)
        slope = float(np.polyfit(ts - ts[0], ds, 1)[0])
        return ProximityReading(d_last, slope)


# ══════════════════════════════════════════════════════════════════════════════
#  Offline
# ══════════════════════════════════════════════════════════════════════════════

def parse_video_position(pos: str) -> float:
    """'MM:SS', 'HH:MM:SS' (fractions allowed) or plain seconds → seconds."""
    secs = 0.0
    for part in pos.strip().split(":"):
        secs = secs * 60 + float(part)
    return secs


def _utc(ts) -> pd.Timestamp:
    """Timestamp in UTC; naive inputs are taken to be UTC."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _decode_wkb_point(hex_str: str) -> tuple[float, float]:
    data = binascii.unhexlify(hex_str)
    fmt = "<" if data[0] == 1 else ">"
    wkb_type = struct.unpack_from(fmt + "I", data, 1)[0]
    offset = 9 if wkb_type & 0x20000000 else 5
    return struct.unpack_from(fmt + "dd", data, offset)


def _lon_lat(df: pd.DataFrame) -> pd.DataFrame:
    """Return float lon/lat in degrees, decoding `geom` or un-scaling 1e7 integers if needed."""
    df = df.copy()
    if "geom" in df.columns and df["geom"].notna().all():
        ll = df["geom"].map(_decode_wkb_point)
        df["lon"], df["lat"] = ll.map(lambda p: p[0]), ll.map(lambda p: p[1])
    elif df["lat"].abs().max() > 90:
        warnings.warn("lon/lat look scaled by 1e7 — dividing. Prefer a log with a `geom` column.")
        df["lon"], df["lat"] = df["lon"] / 1e7, df["lat"] / 1e7
    df["lon"], df["lat"] = df["lon"].astype(float), df["lat"].astype(float)
    return df


class TrajectoryLog:
    """
    Recorded two-vessel log split into 1-second slots counted from the first record.

    Slot k covers [first_record + k s, first_record + k+1 s). Video second k
    (frames k*fps … (k+1)*fps - 1) uses slot k. The video is assumed to start
    one second after the log, so slot k is complete when video second k begins:
    every reading uses only data that would already be available live.

    To keep it causal:
    * a vessel's position in a slot is the mean of its fixes in that slot
      (normally exactly one);
    * a slot where a vessel has no fix holds its last known position for at
      most `max_gap_s` slots, after which there is no reading (→ router fallback);
      nothing is interpolated from later fixes;
    * closure rate is the backward difference d[k] - d[k-1], and 0 when there is
      no previous reading (e.g. the first second).
    """

    def __init__(self, first_record: pd.Timestamp, distance: np.ndarray, closure: np.ndarray,
                 mmsi_a, mmsi_b):
        self.first_record = first_record
        self.distance, self.closure = distance, closure  # NaN = no reading for that slot
        self.mmsi_a, self.mmsi_b = mmsi_a, mmsi_b
        self.times = first_record + pd.to_timedelta(np.arange(len(distance)), unit="s")

    @classmethod
    def from_csv(cls, path, mmsi_a=None, mmsi_b=None, max_gap_s: int = 2) -> "TrajectoryLog":
        df = pd.read_csv(path)
        missing = {"t", "mmsi"} - set(df.columns)
        if missing or not ({"lon", "lat"} <= set(df.columns) or "geom" in df.columns):
            raise ValueError(f"{path}: need columns t, mmsi and lon/lat or geom")
        df = _lon_lat(df)
        df["t"] = pd.to_datetime(df["t"], utc=True, format="mixed")

        ids = list(df["mmsi"].dropna().unique())
        if mmsi_a is None or mmsi_b is None:
            if len(ids) < 2:
                raise ValueError(f"Need two vessels, found MMSI {ids}")
            if len(ids) > 2:
                warnings.warn(f"{len(ids)} MMSIs in log; using {ids[:2]}. Pass mmsi_a/mmsi_b to choose.")
            mmsi_a, mmsi_b = ids[0], ids[1]

        first = df["t"].min()
        df["slot"] = ((df["t"] - first).dt.total_seconds() // 1).astype(int)
        n = int(df["slot"].max()) + 1

        def slots(m):
            tr = df[df["mmsi"] == m]
            if tr.empty:
                raise ValueError(f"No rows for MMSI {m}")
            s = tr.groupby("slot")[["lon", "lat"]].mean().reindex(range(n))
            # hold the last known position for up to max_gap_s slots (causal)
            return s.ffill(limit=max_gap_s)

        sa, sb = slots(mmsi_a), slots(mmsi_b)
        dist = np.full(n, np.nan)
        ok = (sa.notna().all(axis=1) & sb.notna().all(axis=1)).to_numpy()
        if not ok.any():
            raise ValueError("The two vessels never have a position in the same second")
        for k in np.flatnonzero(ok):
            dist[k] = haversine(sa.iat[k, 0], sa.iat[k, 1], sb.iat[k, 0], sb.iat[k, 1])

        # closure rate: backward difference only (causal); 0 without a previous reading
        closure = np.full(n, np.nan)
        closure[1:] = dist[1:] - dist[:-1]
        closure = np.where(np.isnan(closure) & ~np.isnan(dist), 0.0, closure)
        return cls(first, dist, closure, mmsi_a, mmsi_b)

    def slot(self, k: int) -> ProximityReading | None:
        """Reading for second k of the log (k = 0 is the first record's second)."""
        if k < 0 or k >= len(self.distance) or np.isnan(self.distance[k]):
            return None
        return ProximityReading(float(self.distance[k]), float(self.closure[k]))

    def at(self, ts) -> ProximityReading | None:
        """Reading for the slot containing timestamp `ts`, or None."""
        return self.slot(int((_utc(ts) - self.first_record).total_seconds() // 1))

    def for_frame(self, frame_idx: int, fps: float, video_start=None,
                  offset_s: float = 0.0) -> ProximityReading | None:
        """Reading for a video frame.

        Default (video_start=None, offset 0): frame f belongs to second
        floor(f / fps) of the log — frames 0…fps-1 ↔ the first record's second.
        Otherwise log time = video_start + f / fps + offset_s.
        """
        t0 = _utc(video_start) if video_start is not None else self.first_record
        return self.at(t0 + pd.Timedelta(seconds=frame_idx / fps + offset_s))

    def log_time_of_day(self, hms: str) -> pd.Timestamp:
        """'HH:MM:SS[.fff]' on the log's (UTC) date → timestamp."""
        return _utc(f"{self.first_record.date()} {hms}")

    def video_start_from_sync(self, sync: str) -> pd.Timestamp:
        """UTC time of video frame 0 from a 'VIDEO=LOG' pairing, e.g. '01:48=11:54:06'."""
        video_pos, log_hms = sync.split("=", 1)
        return self.log_time_of_day(log_hms) - pd.Timedelta(seconds=parse_video_position(video_pos))

    def coverage(self, video_start, duration_s: float, offset_s: float = 0.0) -> float:
        """Fraction of the video's seconds that have a log reading."""
        n = max(int(np.ceil(duration_s)), 1)
        t0 = _utc(video_start) + pd.Timedelta(seconds=offset_s)
        hits = sum(self.at(t0 + pd.Timedelta(seconds=s)) is not None for s in range(n))
        return hits / n

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"t": self.times, "distance_m": self.distance,
                             "closure_rate_ms": self.closure})
