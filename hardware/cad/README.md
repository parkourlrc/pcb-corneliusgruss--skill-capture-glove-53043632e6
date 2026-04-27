# Glove V3 — printable parts

3MF mesh files for every piece of the assembled V3 glove. Print on any FDM printer that handles 0.2 mm layers in PLA. Print orientation is encoded in the 3MF metadata (open in Bambu Studio, PrusaSlicer, or OrcaSlicer to inspect).

| File | Role |
|---|---|
| `palm_top.3mf` | Top half of the palm shell (cable channels + DAQ pocket) |
| `palm_bottom.3mf` | Bottom half of the palm shell |
| `daq_cover.3mf` | Removable cover over the DAQ PCB pocket |
| `sensor_cover.3mf` | Strain-relief cover for joint-sensor wiring runs |
| `index_proximal.3mf` | Index finger, proximal segment (palm side of MCP joint) |
| `index_distal.3mf` | Index finger, distal segment (fingertip past PIP joint) |
| `3finger_proximal.3mf` | Three-finger module, proximal segment |
| `3finger_distal.3mf` | Three-finger module, distal segment |
| `mcp_pin.3mf` | Hinge pin, MCP joints |
| `pip_pin.3mf` | Hinge pin, PIP joints |
| `pip_joint_retainer.3mf` | Snap clip retaining the PIP pin in the slit-flexure |
| `encoder_cover.3mf` | Magnet seat / cover for each Hall joint sensor PCB |
| `index_pip_sensor_standoff.3mf` | Compliance standoff between index PIP joint and Hall PCB |
| `gopro_mount.3mf` | Strap-on bracket holding the Hero 10 above the wrist |
| `camera_mount_top.3mf` | OV9782 stereo camera holder (top, on-glove V3 build) |
| `camera_mount_bottom.3mf` | OV9782 stereo camera holder (bottom) |
| `camera_assembly_top_cover.3mf` | Cover for the OV9782 top mount |

## Source CAD

The Fusion 360 source archive is not shipped — it's a working file with broken constraints and nonconformant naming, not something an outsider can pick up cleanly. The 3MF files above are what was actually printed and assembled for the V3 build.

Recovering parametric geometry from a 3MF requires re-import + re-construction; the files in this directory are intended as a record of the form factor used to obtain the results in the [paper](../../paper/me740_report.pdf), not as a turnkey replication kit.

## Notes for re-printing

- **PIP slit-flexure compliance.** `index_pip_sensor_standoff.3mf` was tuned (via a thickness sweep, not shipped) to give the Hall sensor's `Bz` axis enough deflection range to recover ~1.3 counts/g. If you reprint with a different filament or layer height the compliance changes, and the per-joint force gain in [`software/configs/calibration/mtcp_sensors.json`](../../software/configs/calibration/mtcp_sensors.json) will need to be re-fit.
- **OV9782 mounts are V3 hardware.** Production SLAM uses the externally-mounted GoPro Hero 10 (`gopro_mount.3mf`). The two on-glove camera mounts are kept here for completeness — they're what's visible in the hero photo.
