"""
Fast checks for adaptive_yolo that need neither a GPU nor the video.

    python -m unittest discover -s tests -v
"""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_yolo import ExitLevel, GradedRouter, LearnedRouter, ProximityReading, TrajectoryLog
from adaptive_yolo.learned import box_iou, count_missed, drop_own_vessel
from adaptive_yolo.model import Detection
from adaptive_yolo.trajectory import parse_video_position

LAT0, LON0 = 37.4397, 24.946


def _geom(lon: float, lat: float) -> str:
    """EWKB point with SRID, as stored in the ASV logs."""
    return (b"\x01" + struct.pack("<I", 0x20000001) + struct.pack("<I", 4326) + struct.pack("<dd", lon, lat)).hex()


def _write_log(path: Path, east_m: list[float], drop_b: set[int] = frozenset()) -> None:
    """Vessel 1 fixed; vessel 2 `east_m[k]` metres east of it in second k (fixes 137 ms later)."""
    t0 = pd.Timestamp("2025-09-17T11:54:06Z")
    rows = []
    for k, d in enumerate(east_m):
        rows.append({"t": (t0 + pd.Timedelta(seconds=k)).isoformat(), "mmsi": 1, "geom": _geom(LON0, LAT0)})
        if k not in drop_b:
            lon = LON0 + d / (111_320 * np.cos(np.radians(LAT0)))
            rows.append({"t": (t0 + pd.Timedelta(seconds=k, milliseconds=137)).isoformat(), "mmsi": 2,
                         "geom": _geom(lon, LAT0)})
    pd.DataFrame(rows).to_csv(path, index=False)


def _det(x1, y1, x2, y2, cls=0, conf=0.9, head=0):
    return Detection(x1, y1, x2, y2, conf, cls, head)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / "log.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_video_position(self):
        self.assertEqual(parse_video_position("01:48"), 108)
        self.assertEqual(parse_video_position("1:00:00"), 3600)
        self.assertEqual(parse_video_position("12.5"), 12.5)

    def test_one_reading_per_second_from_first_record(self):
        _write_log(self.csv, [100, 85, 70])
        log = TrajectoryLog.from_csv(self.csv)
        np.testing.assert_allclose(log.distance, [100, 85, 70], atol=0.2)
        # causal closure rate: 0 in the first second, then backward difference
        np.testing.assert_allclose(log.closure, [0, -15, -15], atol=0.2)
        # frames 0..fps-1 belong to the first record's second
        self.assertAlmostEqual(log.for_frame(29, 30).distance_m, log.distance[0])
        self.assertAlmostEqual(log.for_frame(30, 30).distance_m, log.distance[1])

    def test_missing_fix_held_at_most_two_seconds(self):
        _write_log(self.csv, [100, 90, 80, 70, 60, 50], drop_b={1, 2, 3})
        log = TrajectoryLog.from_csv(self.csv)
        np.testing.assert_allclose(log.distance[:3], [100, 100, 100], atol=0.2)  # held, not interpolated
        self.assertTrue(np.isnan(log.distance[3]))                               # hole longer than 2 s
        self.assertIsNone(log.slot(3))

    def test_sync(self):
        _write_log(self.csv, [100, 85])
        log = TrajectoryLog.from_csv(self.csv)
        start = log.video_start_from_sync("00:10=11:54:06")
        self.assertEqual(start, pd.Timestamp("2025-09-17T11:53:56Z"))


class HeuristicRouterTests(unittest.TestCase):
    def test_binary_rule(self):
        r = GradedRouter(full_dist=30, mid_dist=None, closing_rate=0.5)
        self.assertEqual(r.rule(None), ExitLevel.FULL)
        self.assertEqual(r.rule(ProximityReading(20, -1.0)), ExitLevel.FULL)
        self.assertEqual(r.rule(ProximityReading(20, +0.2)), ExitLevel.P3)   # close but separating
        self.assertEqual(r.rule(ProximityReading(50, -1.0)), ExitLevel.P3)

    def test_graded_rule(self):
        r = GradedRouter(full_dist=30, mid_dist=60, closing_rate=0.5)
        self.assertEqual(r.rule(ProximityReading(20, +0.2)), ExitLevel.P3_P4)
        self.assertEqual(r.rule(ProximityReading(80, -1.0)), ExitLevel.P3)

    def test_hysteresis_delays_stepping_down(self):
        r = GradedRouter(full_dist=30, mid_dist=None, closing_rate=0.5, hold_frames=3)
        close, far = ProximityReading(10, -1.0), ProximityReading(100, 0.0)
        levels = [r(close)] + [r(far) for _ in range(4)]
        self.assertEqual(levels, [ExitLevel.FULL] * 3 + [ExitLevel.P3] * 2)


class MatchingTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(box_iou(_det(0, 0, 10, 10), _det(0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(box_iou(_det(0, 0, 10, 10), _det(5, 0, 15, 10)), 1 / 3)

    def test_count_missed_requires_same_class(self):
        full = [_det(0, 0, 10, 10, cls=0), _det(50, 50, 60, 60, cls=1)]
        self.assertEqual(count_missed(full, [_det(0, 0, 10, 10, cls=0)]), (2, 1))
        self.assertEqual(count_missed(full, [_det(0, 0, 10, 10, cls=1)]), (2, 2))

    def test_own_vessel_dropped(self):
        dets = [_det(0, 1500, 100, 2160), _det(0, 0, 10, 10)]
        self.assertEqual(len(drop_own_vessel(dets, 2160, 0.97)), 1)
        self.assertEqual(len(drop_own_vessel(dets, 2160, None)), 2)


class LearnedRouterTests(unittest.TestCase):
    def test_cold_start_runs_full(self):
        r = LearnedRouter(target_miss=0.3, probe_rate=0.0)
        self.assertEqual(r(ProximityReading(50, 0.0)), ExitLevel.FULL)
        self.assertEqual(r(None), ExitLevel.FULL)

    def test_learns_that_p3_is_safe_when_far(self):
        r = LearnedRouter(target_miss=0.3, probe_rate=0.0)
        far = ProximityReading(80, 0.0)
        full = [_det(0, 0, 10, 10)]
        no_miss = {ExitLevel.FULL: full, ExitLevel.P3_P4: full, ExitLevel.P3: full}
        for _ in range(200):          # FULL frames on which P3 found everything
            r(far)
            r.observe(no_miss)
        self.assertEqual(r(far), ExitLevel.P3)

    def test_deterministic_given_seed(self):
        readings = [ProximityReading(d, -0.5) for d in np.linspace(60, 5, 50)]
        a, b = (LearnedRouter(strategy="thompson", probe_rate=0.1, seed=3) for _ in range(2))
        self.assertEqual([a(x) for x in readings], [b(x) for x in readings])


if __name__ == "__main__":
    unittest.main()
