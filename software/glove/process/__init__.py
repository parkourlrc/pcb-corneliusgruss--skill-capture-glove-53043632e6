"""Offline processing: raw sessions → training-ready data.

Pipeline stages live in submodules:
- slam/   — ORB-SLAM3 map build + localize (via UMI's gopro_slam docker)
- sync/   — GoPro wall-clock ↔ sensor stream alignment
- align/  — per-frame pose + proprio into episode.parquet
- export/ — sessions → zarr replay buffer for training
"""
