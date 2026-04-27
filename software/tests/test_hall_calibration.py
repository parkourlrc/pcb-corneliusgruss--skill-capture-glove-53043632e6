"""Tests for Hall sensor calibration module."""

import json
from pathlib import Path

import numpy as np
import pytest

from glove.data.hall_calibration import (
    SENS_XY,
    SENS_Z,
    HallCalibration,
    HallReading,
    SensorConfig,
    calibrate_offsets,
    generate_synthetic_hall_data,
    lsb_to_ut,
    process_hall_frame,
)


# ============================================================
# Sensitivity Lookup
# ============================================================


class TestLsbToUt:
    def test_known_value_xy(self):
        # GAIN_SEL=0, RES=0: 0.751 µT/LSB (datasheet Table 17, T=25°C)
        assert lsb_to_ut(1000, gain_sel=0, res=0, axis="xy") == pytest.approx(751.0)

    def test_known_value_z(self):
        # GAIN_SEL=0, RES=0: 1.210 µT/LSB (datasheet Table 17, T=25°C)
        assert lsb_to_ut(1000, gain_sel=0, res=0, axis="z") == pytest.approx(1210.0)

    def test_z_higher_than_xy(self):
        for key in SENS_XY:
            assert SENS_Z[key] > SENS_XY[key]

    def test_negative_value(self):
        result = lsb_to_ut(-100, gain_sel=5, res=0, axis="xy")
        assert result < 0

    def test_zero(self):
        assert lsb_to_ut(0, gain_sel=3, res=1, axis="xy") == 0.0

    def test_invalid_gain_raises(self):
        with pytest.raises(ValueError, match="gain_sel"):
            lsb_to_ut(100, gain_sel=8, res=0)

    def test_invalid_res_raises(self):
        with pytest.raises(ValueError, match="res"):
            lsb_to_ut(100, gain_sel=0, res=4)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis"):
            lsb_to_ut(100, gain_sel=0, res=0, axis="w")


# ============================================================
# Angle Computation
# ============================================================


class TestAngleComputation:
    """Test angle computation via process_hall_frame with known inputs."""

    def _make_calibration(self, offset_ut=None, reference_angle_deg=0.0):
        """Helper: calibration with zero offsets by default."""
        if offset_ut is None:
            offset_ut = np.zeros(3)
        sensors = [
            SensorConfig(
                name=f"sensor_{i}",
                gain_sel=0,
                res_xy=0,
                res_z=0,
                offset_ut=offset_ut.copy(),
                reference_angle_deg=reference_angle_deg,
            )
            for i in range(4)
        ]
        return HallCalibration(sensors=sensors)

    def _raw_from_angle(self, angle_deg: float, b_max_lsb: int = 1000) -> tuple:
        """Helper: generate raw LSB tuple for a given angle."""
        angle_rad = np.radians(angle_deg)
        bx = int(round(b_max_lsb * np.cos(angle_rad)))
        by = int(round(b_max_lsb * np.sin(angle_rad)))
        return (bx, by, 0)

    def test_zero_angle(self):
        cal = self._make_calibration()
        raw = tuple(self._raw_from_angle(0.0) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        np.testing.assert_allclose(reading.angles_deg, 0.0, atol=0.1)

    def test_ninety_degrees(self):
        cal = self._make_calibration()
        raw = tuple(self._raw_from_angle(90.0) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        np.testing.assert_allclose(reading.angles_deg, 90.0, atol=0.1)

    def test_180_degrees(self):
        cal = self._make_calibration()
        raw = tuple(self._raw_from_angle(180.0) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        np.testing.assert_allclose(reading.angles_deg, 180.0, atol=0.1)

    def test_270_degrees(self):
        cal = self._make_calibration()
        raw = tuple(self._raw_from_angle(270.0) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        np.testing.assert_allclose(reading.angles_deg, 270.0, atol=0.1)

    def test_reference_angle_offset(self):
        cal = self._make_calibration(reference_angle_deg=45.0)
        raw = tuple(self._raw_from_angle(0.0) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        np.testing.assert_allclose(reading.angles_deg, 45.0, atol=0.1)

    def test_magnitude_computed(self):
        cal = self._make_calibration()
        raw = tuple(self._raw_from_angle(45.0, b_max_lsb=1000) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        # magnitude should be ~b_max_lsb * sensitivity
        expected_mag = 1000 * SENS_XY[(0, 0)]
        np.testing.assert_allclose(reading.magnitudes_ut, expected_mag, rtol=0.02)

    def test_bz_passthrough(self):
        cal = self._make_calibration()
        bz_lsb = 500
        raw = tuple((1000, 0, bz_lsb) for _ in range(4))
        reading = process_hall_frame(raw, cal)
        expected_bz = bz_lsb * SENS_Z[(0, 0)]
        np.testing.assert_allclose(reading.bz_ut, expected_bz, atol=0.1)


# ============================================================
# Calibration
# ============================================================


class TestCalibrateOffsets:
    def test_static_calibration(self):
        """Calibrate from static readings at a known angle, then verify processing."""
        rng = np.random.default_rng(0)
        gain_sel, res_xy, res_z = 5, 0, 0
        sens_xy = SENS_XY[(gain_sel, res_xy)]
        sens_z = SENS_Z[(gain_sel, res_z)]
        b_max_ut = 8500.0
        rest_angle_deg = 45.0
        rest_angle_rad = np.radians(rest_angle_deg)

        # Generate 50 static samples at rest_angle with small noise
        n = 50
        samples = np.empty((n, 4, 3), dtype=np.int16)
        for s in range(4):
            bx_ut = b_max_ut * np.cos(rest_angle_rad) + rng.normal(0, 10, n)
            by_ut = b_max_ut * np.sin(rest_angle_rad) + rng.normal(0, 10, n)
            bz_ut = rng.normal(0, 10, n)
            samples[:, s, 0] = np.round(bx_ut / sens_xy).astype(np.int16)
            samples[:, s, 1] = np.round(by_ut / sens_xy).astype(np.int16)
            samples[:, s, 2] = np.round(bz_ut / sens_z).astype(np.int16)

        cal = calibrate_offsets(
            samples, gain_sel=gain_sel, res_xy=res_xy, res_z=res_z,
            reference_angles_deg=np.full(4, rest_angle_deg),
        )
        # Process a frame at the same rest angle — should get rest_angle_deg back
        raw = tuple(
            (int(samples[0, s, 0]), int(samples[0, s, 1]), int(samples[0, s, 2]))
            for s in range(4)
        )
        reading = process_hall_frame(raw, cal)
        np.testing.assert_allclose(reading.angles_deg, rest_angle_deg, atol=1.0)

        # Process a frame at 90° — should get 90° back
        angle_90 = np.radians(90.0)
        raw_90 = tuple(
            (int(round(b_max_ut * np.cos(angle_90) / sens_xy)),
             int(round(b_max_ut * np.sin(angle_90) / sens_xy)),
             0)
            for _ in range(4)
        )
        reading_90 = process_hall_frame(raw_90, cal)
        np.testing.assert_allclose(reading_90.angles_deg, 90.0, atol=1.0)

    def test_single_sample(self):
        raw_lsb, _, _ = generate_synthetic_hall_data(num_frames=1, seed=7)
        cal = calibrate_offsets(raw_lsb, gain_sel=5, res_xy=0, res_z=0)
        assert cal.num_samples == 1
        assert len(cal.sensors) == 4

    def test_reference_angles_applied(self):
        """Verify that reference_angles_deg shifts the stored offset correctly."""
        raw_lsb, _, _ = generate_synthetic_hall_data(num_frames=10, seed=3)
        ref_a = np.array([0.0, 0.0, 0.0, 0.0])
        ref_b = np.array([10.0, 20.0, 30.0, 40.0])
        cal_a = calibrate_offsets(raw_lsb, gain_sel=5, res_xy=0, res_z=0,
                                  reference_angles_deg=ref_a)
        cal_b = calibrate_offsets(raw_lsb, gain_sel=5, res_xy=0, res_z=0,
                                  reference_angles_deg=ref_b)
        for i in range(4):
            diff = cal_b.sensors[i].reference_angle_deg - cal_a.sensors[i].reference_angle_deg
            assert abs(diff - (ref_b[i] - ref_a[i])) < 0.01

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            calibrate_offsets(np.zeros((10, 3)), gain_sel=5, res_xy=0, res_z=0)

    def test_timestamp_recorded(self):
        raw_lsb, _, _ = generate_synthetic_hall_data(num_frames=10, seed=1)
        cal = calibrate_offsets(raw_lsb, gain_sel=5, res_xy=0, res_z=0)
        assert cal.calibrated_at != ""


# ============================================================
# Round-Trip
# ============================================================


class TestRoundTrip:
    def test_synthetic_calibrate_process(self):
        """End-to-end: generate data → use first frame as rest → calibrate → process."""
        gain_sel, res_xy, res_z = 5, 0, 0

        raw_lsb, true_angles, _ = generate_synthetic_hall_data(
            num_frames=300, seed=42, noise_ut=20.0,
            gain_sel=gain_sel, res_xy=res_xy, res_z=res_z,
        )

        # Calibrate using first frame as "rest" at its true angle
        rest_angles = true_angles[0]
        cal = calibrate_offsets(
            raw_lsb[:1], gain_sel=gain_sel, res_xy=res_xy, res_z=res_z,
            reference_angles_deg=rest_angles,
        )

        # Process all frames and check angle recovery
        angle_errors = []
        for t in range(1, 300):
            hall_raw = tuple(
                (int(raw_lsb[t, s, 0]), int(raw_lsb[t, s, 1]), int(raw_lsb[t, s, 2]))
                for s in range(4)
            )
            reading = process_hall_frame(hall_raw, cal)
            for s in range(4):
                diff = reading.angles_deg[s] - (true_angles[t, s] % 360)
                diff = (diff + 180) % 360 - 180
                angle_errors.append(abs(diff))

        rmse = np.sqrt(np.mean(np.array(angle_errors) ** 2))
        assert rmse < 5.0, f"Angle RMSE {rmse:.2f}° exceeds 5° threshold"


# ============================================================
# Serialization
# ============================================================


class TestSerialization:
    def test_sensor_config_round_trip(self):
        cfg = SensorConfig(
            name="test", gain_sel=3, res_xy=1, res_z=0,
            offset_ut=np.array([1.0, 2.0, 3.0]),
            reference_angle_deg=45.0,
        )
        restored = SensorConfig.from_dict(cfg.to_dict())
        assert restored.name == cfg.name
        assert restored.gain_sel == cfg.gain_sel
        np.testing.assert_array_equal(restored.offset_ut, cfg.offset_ut)
        assert restored.reference_angle_deg == cfg.reference_angle_deg

    def test_hall_calibration_round_trip(self):
        raw_lsb, _, _ = generate_synthetic_hall_data(num_frames=10, seed=5)
        cal = calibrate_offsets(raw_lsb, gain_sel=5, res_xy=0, res_z=0)
        restored = HallCalibration.from_dict(cal.to_dict())
        assert len(restored.sensors) == 4
        for i in range(4):
            np.testing.assert_array_almost_equal(
                restored.sensors[i].offset_ut, cal.sensors[i].offset_ut
            )

    def test_save_load_json(self, tmp_path):
        raw_lsb, _, _ = generate_synthetic_hall_data(num_frames=10, seed=9)
        cal = calibrate_offsets(raw_lsb, gain_sel=5, res_xy=0, res_z=0)

        path = tmp_path / "hall_cal.json"
        cal.save(path)
        loaded = HallCalibration.load(path)

        assert loaded.num_samples == cal.num_samples
        for i in range(4):
            np.testing.assert_array_almost_equal(
                loaded.sensors[i].offset_ut, cal.sensors[i].offset_ut
            )

    def test_hall_reading_to_array(self):
        reading = HallReading(
            angles_deg=np.array([10.0, 20.0, 30.0, 40.0]),
            magnitudes_ut=np.array([100.0, 200.0, 300.0, 400.0]),
            bz_ut=np.array([50.0, 60.0, 70.0, 80.0]),
        )
        arr = reading.to_array()
        assert arr.shape == (12,)
        assert arr[0] == 10.0   # angle_0
        assert arr[1] == 100.0  # mag_0
        assert arr[2] == 50.0   # bz_0

    def test_hall_reading_to_angle_array(self):
        reading = HallReading(
            angles_deg=np.array([10.0, 20.0, 30.0, 40.0]),
            magnitudes_ut=np.array([100.0, 200.0, 300.0, 400.0]),
            bz_ut=np.array([50.0, 60.0, 70.0, 80.0]),
        )
        arr = reading.to_angle_array()
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0, 40.0])


# ============================================================
# Synthetic Generator
# ============================================================


class TestSyntheticGenerator:
    def test_deterministic_seed(self):
        r1 = generate_synthetic_hall_data(seed=123)
        r2 = generate_synthetic_hall_data(seed=123)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_output_shapes(self):
        raw, angles, offsets = generate_synthetic_hall_data(num_frames=100)
        assert raw.shape == (100, 4, 3)
        assert angles.shape == (100, 4)
        assert offsets.shape == (4, 3)

    def test_raw_dtype_int16(self):
        raw, _, _ = generate_synthetic_hall_data()
        assert raw.dtype == np.int16

    def test_lsb_in_int16_range(self):
        raw, _, _ = generate_synthetic_hall_data()
        assert raw.min() >= -32768
        assert raw.max() <= 32767

    def test_angles_in_reasonable_range(self):
        _, angles, _ = generate_synthetic_hall_data(num_frames=500, seed=0)
        # center_deg in [20, 90], amplitude in [10, 45] → range roughly [-25, 135]
        assert angles.min() >= -30.0
        assert angles.max() <= 150.0
