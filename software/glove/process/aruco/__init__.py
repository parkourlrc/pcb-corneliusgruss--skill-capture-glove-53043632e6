"""ArUco detection + tag-frame calibration, ported from UMI.

Two CLI entry points:

    glove.process detect-aruco   <session>        # writes aruco/tag_detection.pkl
    glove.process calibrate-tag  <mapping>        # writes slam/tx_slam_tag.json

`align.py` consumes the mapping session's `tx_slam_tag.json` to express demo
trajectories in the tag frame (gravity-aligned by construction).
"""
