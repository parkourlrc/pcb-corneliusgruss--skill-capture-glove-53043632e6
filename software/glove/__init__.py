"""Skill-Capture Glove — capture, alignment, and policy training.

Captures human manipulation demonstrations from a worn glove (4× MLX90393
Hall joint sensors, 2× VL53L1X ToF, GoPro Hero 10 SLAM camera), aligns them
to a SLAM trajectory in a gravity-aligned world frame, and trains an
ACT-class imitation policy on the result.
"""

__version__ = "0.1.0"
