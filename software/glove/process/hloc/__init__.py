"""HLoc-based mapping + localization + smoothing.

Replaces ORB-SLAM3's `--load_map` relocalization (community-known to plateau at
~20% pass rate on non-Sunday hardware — see `project_umi_slam_ceiling` memory)
with the SuperPoint + NetVLAD + LightGlue + COLMAP stack from
[Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization),
plus a pose-graph smoother that turns the per-frame PnP measurements into a
metric trajectory comparable in quality to ORB-SLAM3's clean runs.

Pipeline:

    build-model    mapping MP4 → SfM model (COLMAP + tag-based scale)
    localize-hloc  demo MP4 → smoothed camera_trajectory.csv (same schema as ORB-SLAM3)

Downstream stages (`detect-aruco`, `calibrate-tag`, `align`) are unchanged.
"""
