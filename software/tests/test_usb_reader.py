"""Tests for STM32 DAQ binary USB reader."""

import struct
from unittest.mock import MagicMock, patch

import pytest

from glove.data.usb_reader import (
    ACCEL_LSB_TO_MS2,
    GYRO_LSB_TO_RADS,
    PACKET_IMU,
    PACKET_PROPRIO,
    PACKET_SIZES,
    DaqReader,
    ImuFrame,
    ProprioFrame,
    parse_imu,
    parse_proprio,
)


# --- Helpers ---


def build_imu_packet(
    timestamp_us: int = 1000000,
    accel: tuple[int, int, int] = (100, -200, 16384),
    gyro: tuple[int, int, int] = (50, -50, 10),
) -> bytes:
    """Build a raw 17-byte IMU packet."""
    return struct.pack(
        "<BI3h3h",
        PACKET_IMU,
        timestamp_us,
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2],
    )


def build_proprio_packet(
    timestamp_us: int = 2000000,
    encoders: tuple[int, ...] = (1024, 2048, 3072, 4095),
    hall: tuple[int, ...] = tuple(range(12)),
    tof: tuple[int, int] = (150, 0xFFFF),
    sync_ts: int = 0,
    flags: int = 0x03,
    sync_seq: int = 0,
) -> bytes:
    """Build a raw 48-byte proprioceptive packet."""
    return struct.pack(
        "<BI4H12h2HIBH",
        PACKET_PROPRIO,
        timestamp_us,
        *encoders,
        *hall,
        *tof,
        sync_ts,
        flags,
        sync_seq,
    )


# --- Fixtures ---


@pytest.fixture
def imu_bytes():
    return build_imu_packet()


@pytest.fixture
def proprio_bytes():
    return build_proprio_packet()


# --- ImuFrame parsing ---


class TestImuParsing:
    def test_basic_parse(self, imu_bytes):
        frame = parse_imu(imu_bytes, local_time=99.0)

        assert isinstance(frame, ImuFrame)
        assert frame.timestamp_us == 1000000
        assert frame.accel_raw == (100, -200, 16384)
        assert frame.gyro_raw == (50, -50, 10)
        assert frame.local_time == 99.0

    def test_accel_conversion(self):
        packet = build_imu_packet(accel=(1000, 0, 0))
        frame = parse_imu(packet)

        expected = 1000 * ACCEL_LSB_TO_MS2
        assert abs(frame.accel_ms2[0] - expected) < 1e-6
        assert frame.accel_ms2[1] == 0.0
        assert frame.accel_ms2[2] == 0.0

    def test_gyro_conversion(self):
        packet = build_imu_packet(gyro=(0, 1000, 0))
        frame = parse_imu(packet)

        expected = 1000 * GYRO_LSB_TO_RADS
        assert abs(frame.gyro_rads[1] - expected) < 1e-6

    def test_packet_size(self):
        assert len(build_imu_packet()) == PACKET_SIZES[PACKET_IMU]
        assert PACKET_SIZES[PACKET_IMU] == 17

    def test_wrong_type_byte(self):
        bad = b"\x02" + b"\x00" * 16
        with pytest.raises(ValueError, match="Expected packet type 0x01"):
            parse_imu(bad)

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="must be 17 bytes"):
            parse_imu(b"\x01\x00\x00")

    def test_to_dict(self, imu_bytes):
        frame = parse_imu(imu_bytes, local_time=1.0)
        d = frame.to_dict()

        assert "timestamp_us" in d
        assert "accel_x" in d
        assert "gyro_z" in d
        assert d["local_time"] == 1.0

    def test_negative_values(self):
        packet = build_imu_packet(accel=(-32768, 32767, 0), gyro=(-1, 1, -32768))
        frame = parse_imu(packet)

        assert frame.accel_raw[0] == -32768
        assert frame.accel_raw[1] == 32767
        assert frame.gyro_raw[2] == -32768


# --- ProprioFrame parsing ---


class TestProprioParsing:
    def test_basic_parse(self, proprio_bytes):
        frame = parse_proprio(proprio_bytes, local_time=42.0)

        assert isinstance(frame, ProprioFrame)
        assert frame.timestamp_us == 2000000
        assert frame.encoder_raw == (1024, 2048, 3072, 4095)
        assert frame.local_time == 42.0

    def test_hall_sensor_grouping(self, proprio_bytes):
        frame = parse_proprio(proprio_bytes)

        # Hall data is range(12) -> 4 sensors x 3 axes
        assert frame.hall_raw[0] == (0, 1, 2)
        assert frame.hall_raw[1] == (3, 4, 5)
        assert frame.hall_raw[2] == (6, 7, 8)
        assert frame.hall_raw[3] == (9, 10, 11)

    def test_tof_valid_reading(self):
        packet = build_proprio_packet(tof=(250, 300))
        frame = parse_proprio(packet)

        assert frame.tof_mm == (250, 300)

    def test_tof_0xffff_becomes_minus_one(self):
        packet = build_proprio_packet(tof=(0xFFFF, 100))
        frame = parse_proprio(packet)

        assert frame.tof_mm[0] == -1
        assert frame.tof_mm[1] == 100

    def test_tof_both_invalid(self):
        packet = build_proprio_packet(tof=(0xFFFF, 0xFFFF))
        frame = parse_proprio(packet)

        assert frame.tof_mm == (-1, -1)

    def test_encoder_degrees(self):
        packet = build_proprio_packet(encoders=(0, 1024, 2048, 4096))
        frame = parse_proprio(packet)

        # 0 -> 0.0, 1024 -> 90.0, 2048 -> 180.0
        degs = frame.encoder_degrees
        assert abs(degs[0] - 0.0) < 0.1
        assert abs(degs[1] - 90.0) < 0.1
        assert abs(degs[2] - 180.0) < 0.1

    def test_sync_timestamp(self):
        packet = build_proprio_packet(sync_ts=5000000)
        frame = parse_proprio(packet)

        assert frame.sync_timestamp_us == 5000000

    def test_sync_zero_means_no_sync(self):
        packet = build_proprio_packet(sync_ts=0)
        frame = parse_proprio(packet)

        assert frame.sync_timestamp_us == 0

    def test_packet_size(self):
        assert len(build_proprio_packet()) == PACKET_SIZES[PACKET_PROPRIO]
        assert PACKET_SIZES[PACKET_PROPRIO] == 48

    def test_wrong_type_byte(self):
        bad = b"\x01" + b"\x00" * 47
        with pytest.raises(ValueError, match="Expected packet type 0x02"):
            parse_proprio(bad)

    def test_sync_seq(self):
        packet = build_proprio_packet(sync_seq=42)
        frame = parse_proprio(packet)

        assert frame.sync_seq == 42

    def test_sync_seq_wraps_at_65535(self):
        packet = build_proprio_packet(sync_seq=65535)
        frame = parse_proprio(packet)

        assert frame.sync_seq == 65535

    def test_to_dict(self, proprio_bytes):
        frame = parse_proprio(proprio_bytes)
        d = frame.to_dict()

        assert "index_mcp" in d
        assert "hall_0" in d
        assert "tof_top_mm" in d
        assert "status_flags" in d
        assert "sync_seq" in d


# --- Status flags ---


class TestStatusFlags:
    def test_recording_flag(self):
        packet = build_proprio_packet(flags=0x01)
        frame = parse_proprio(packet)

        assert frame.recording is True
        assert frame.sync_fired is False
        assert frame.btn_pressed is False

    def test_sync_fired_flag(self):
        packet = build_proprio_packet(flags=0x02)
        frame = parse_proprio(packet)

        assert frame.recording is False
        assert frame.sync_fired is True

    def test_btn_pressed_flag(self):
        packet = build_proprio_packet(flags=0x04)
        frame = parse_proprio(packet)

        assert frame.btn_pressed is True

    def test_all_flags(self):
        packet = build_proprio_packet(flags=0x07)
        frame = parse_proprio(packet)

        assert frame.recording is True
        assert frame.sync_fired is True
        assert frame.btn_pressed is True

    def test_no_flags(self):
        packet = build_proprio_packet(flags=0x00)
        frame = parse_proprio(packet)

        assert frame.recording is False
        assert frame.sync_fired is False
        assert frame.btn_pressed is False


# --- Packet type detection ---


class TestPacketTypeDetection:
    def test_imu_type_byte(self):
        packet = build_imu_packet()
        assert packet[0] == 0x01

    def test_proprio_type_byte(self):
        packet = build_proprio_packet()
        assert packet[0] == 0x02


# --- DaqReader with mocked serial ---


class TestDaqReaderResync:
    """Test DaqReader.read_frame() with garbage bytes before valid packets."""

    def _make_reader(self, data: bytes) -> DaqReader:
        """Create a DaqReader with a mocked serial port returning given data."""
        reader = DaqReader(port="/dev/fake")
        mock_serial = MagicMock()

        # Track position in the data buffer
        pos = [0]

        def mock_read(n):
            start = pos[0]
            end = min(start + n, len(data))
            pos[0] = end
            return data[start:end]

        mock_serial.read = mock_read
        mock_serial.is_open = True
        reader.serial = mock_serial
        return reader

    def test_clean_imu_read(self):
        packet = build_imu_packet()
        reader = self._make_reader(packet)

        frame = reader.read_frame()

        assert isinstance(frame, ImuFrame)
        assert frame.timestamp_us == 1000000
        assert reader._resync_count == 0

    def test_clean_proprio_read(self):
        packet = build_proprio_packet()
        reader = self._make_reader(packet)

        frame = reader.read_frame()

        assert isinstance(frame, ProprioFrame)
        assert frame.timestamp_us == 2000000

    def test_garbage_before_imu(self):
        garbage = bytes([0xFF, 0xAA, 0x00, 0x55, 0x03])
        packet = build_imu_packet(timestamp_us=12345)
        reader = self._make_reader(garbage + packet)

        frame = reader.read_frame()

        assert isinstance(frame, ImuFrame)
        assert frame.timestamp_us == 12345
        assert reader._resync_count == len(garbage)

    def test_garbage_before_proprio(self):
        garbage = bytes([0x99, 0x88])
        packet = build_proprio_packet(timestamp_us=54321)
        reader = self._make_reader(garbage + packet)

        frame = reader.read_frame()

        assert isinstance(frame, ProprioFrame)
        assert frame.timestamp_us == 54321
        assert reader._resync_count == 2

    def test_multiple_frames(self):
        imu = build_imu_packet(timestamp_us=100)
        proprio = build_proprio_packet(timestamp_us=200)
        reader = self._make_reader(imu + proprio)

        f1 = reader.read_frame()
        f2 = reader.read_frame()

        assert isinstance(f1, ImuFrame)
        assert f1.timestamp_us == 100
        assert isinstance(f2, ProprioFrame)
        assert f2.timestamp_us == 200

    def test_timeout_returns_none(self):
        reader = self._make_reader(b"")

        frame = reader.read_frame()
        assert frame is None

    def test_incomplete_packet_returns_none(self):
        # Only type byte + 3 bytes (not enough for IMU)
        reader = self._make_reader(b"\x01\x00\x00\x00")

        frame = reader.read_frame()
        assert frame is None

    def test_no_serial_returns_none(self):
        reader = DaqReader(port="/dev/fake")
        reader.serial = None

        assert reader.read_frame() is None


# --- Context manager ---


class TestDaqReaderContextManager:
    @patch("glove.data.usb_reader.serial.Serial")
    def test_context_manager(self, mock_serial_cls):
        mock_instance = MagicMock()
        mock_serial_cls.return_value = mock_instance

        with DaqReader(port="/dev/fake") as daq:
            assert daq.serial is not None

        mock_instance.close.assert_called_once()
