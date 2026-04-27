# Reproducing the pipeline

End-to-end commands from a fresh clone to a trained ACT policy. Hardware build assumes you have the [DAQ PCB](../hardware/daq_pcb/) and [joint sensor PCBs](../hardware/joint_sensor_pcb/) manufactured, the V3 glove printed and assembled, and a GoPro Hero 10 with the MaxLens Mod 1.0.

## 1. Software install

```bash
git clone https://github.com/corneliusgruss/skill-capture-glove.git
cd skill-capture-glove/software
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,daq,dashboard]"
```

External dependencies handled outside `pip`:

- **Docker Desktop** — required by the SLAM stack. Pulls `chicheng/orb_slam3:latest` and `chicheng/openicc:latest` automatically on first use.
- **GoPro Labs firmware** on the Hero 10 ([install guide](https://gopro.github.io/labs/)).

## 2. Firmware flash

```bash
cd ../firmware/stm32_daq
mkdir -p lib && cd lib
git clone https://github.com/STMicroelectronics/STM32CubeG4.git
git clone https://github.com/hathach/tinyusb.git
cd ..

mkdir build && cd build
cmake -DBLINK_TEST=OFF .. && make -j8
dfu-util -a 0 -s 0x08000000:leave -D glove_daq.bin
```

## 3. Capture a session

Once per session, sync the GoPro's clock to UTC by pointing it at the [Labs precision-time QR](https://gopro.github.io/labs/control/precisiontime_utc/) for ~3 s.

```bash
# Continuous DAQ writer — leave running for the whole capture day.
glove-capture record

# In another terminal, for each demo:
#   1. Press record on the GoPro
#   2. Perform the task
#   3. Press stop on the GoPro
# At end of day: Ctrl-C the recorder.

# Offload MP4s from SD card to disk before ingest.
glove-capture offload
```

## 4. Process raw clips into sessions

Each MP4 plus the matching slice of the master tape becomes one session directory.

```bash
# Mapping clip (run once per workspace; covers the demo area from many angles).
glove-process ingest --mp4 data/raw_video/2026-04-25/GX010001.MP4 \
                     --stream-dir data/raw_stream/2026-04-25/ \
                     --task mug_coaster --kind mapping

# Demo clips (one per task instance).
glove-process ingest --mp4 data/raw_video/2026-04-25/GX010002.MP4 \
                     --stream-dir data/raw_stream/2026-04-25/ \
                     --task mug_coaster
```

## 5. Build the SLAM map

Run once per workspace. The mapping clip must start with the ArUco tag fully visible and unoccluded for ~2 s (initialization race; otherwise `gopro_slam` segfaults on `--save_map`).

```bash
MAP=data/sessions/map_2026-04-25_mug_coaster_0001
glove-process extract-gpmf $MAP
glove-process build-map    $MAP
glove-process detect-aruco $MAP
glove-process calibrate-tag $MAP   # writes slam/tx_slam_tag.json (gravity-aligned)
```

`build-map` writes a `SlamStats` verdict (CLEAN / NOISY / RAILS) and `map_quality` block alongside the trajectory.

## 6. Localize and align each demo

```bash
DEMO=data/sessions/demo_2026-04-25_mug_coaster_0002
glove-process extract-gpmf $DEMO
glove-process localize     $DEMO --map $MAP
glove-process align        $DEMO        # writes aligned/episode.parquet
```

`align` produces the training-ready `episode.parquet`: per-frame MTCP pose in the gravity-aligned tag frame plus interpolated Hall and ToF readings. Column-level schema is documented inline at the top of [`software/glove/process/align.py`](../software/glove/process/align.py).

## 7. Inspect and review

Launch the dashboard against the session directory to scrub through demos, see per-session quality verdicts, and approve / reject for inclusion in the training manifest:

```bash
glove-dashboard --data-dir data/sessions
# open http://localhost:8000
```

The dashboard reads `aligned/episode.parquet` directly — no SLAM rerun required.

## 8. Build a dataset and train

```bash
# Pick demos to include and write a manifest.
python -m glove.scripts.build_dataset \
    --sessions data/sessions/demo_*_mug_coaster_* \
    --out data/datasets/mug_coaster_v1

# Train ACT (frozen DINOv2 backbone, 23-D obs, 21-D action).
glove-policy train --config ../software/configs/act.yaml \
                   --dataset data/datasets/mug_coaster_v1/manifest.yaml
```

Training was run on Boston University's SCC (single L40S). The production checkpoint reached **28.8 mm position L1, 6.31° rotation L1, 736 LSB Hall L1** on a 27-session held-out validation set, scored against the HLoc + σ=5 ground truth. The training set is 182 demonstrations: 132 surviving a label-quality audit (23% of the original 172 candidates excluded for silent SLAM pose-graph re-anchoring artifacts) plus 50 clean demonstrations recorded the following day. Best checkpoint at epoch 38 of a 150-epoch budget. See §VII of the paper for the full configuration.

## 9. Validate against ground truth (optional)

The headline SLAM result in the README — HLoc within 0.2 mm of the absolute camera + PnP physical noise floor — comes from the AprilGrid validation script:

```bash
python -m glove.scripts.pose_validation_aprilgrid \
    --session $DEMO \
    --target  ../software/configs/calibration/kalibr/target.yaml
```
