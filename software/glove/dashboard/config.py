"""Dashboard configuration."""

from pathlib import Path

# Data directory (sessions stored here)
DATA_DIR = Path("data/sessions")

# Web server
WEB_HOST = "0.0.0.0"
WEB_PORT = 8420

# Camera settings (indices auto-detected at runtime, these are fallbacks)
CAMERA_LEFT_INDEX = 1
CAMERA_RIGHT_INDEX = 2
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800
CAMERA_FPS = 100  # Set high so AVFoundation doesn't limit hardware trigger rate

# Recording
RECORD_POLL_MS = 2   # How often to check for new stereo pairs (ms)

# Streaming rates
STREAM_FPS = 20  # WebSocket camera stream
SENSOR_HZ = 10   # WebSocket sensor stream

# Camera trigger (OV9782 external trigger via F pad + ESP32 GPIO25)
# On Linux/V4L2: sets CAP_PROP_TRIGGER to enable driver-level trigger mode
# On macOS/AVFoundation: no-op (F pad trigger works at sensor level)
CAMERA_TRIGGER_MODE = False

# Sensor backend: "auto" detects STM32 first then ESP32, or force with "esp32"/"stm32"
SENSOR_BACKEND = "auto"

# ESP32 serial
ESP32_BAUDRATE = 460800  # 200Hz IMU needs >132500 baud; 460800 gives 3.5x headroom

# Session database
DB_NAME = "sessions.db"

# Quality auto-gating thresholds
QUALITY_AUTO_APPROVE = 0.7
QUALITY_AUTO_REJECT = 0.3
