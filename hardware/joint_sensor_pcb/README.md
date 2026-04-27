# Joint sensor PCB

Micro-PCB that mounts on each finger flexure to capture joint angle and axial compliance from a single diametric magnet.

| | |
|---|---|
| **Sensor** | Melexis MLX90393 (3-axis Hall, QFN-16) |
| **Interface** | I²C @ 0x0C (routed through DAQ board's TCA9548A mux) |
| **Power** | 3.3 V from DAQ board |
| **Form factor** | 15 × 20 mm, 2-layer FR4 |
| **Connector** | JST-SH 4-pin right-angle (3V3 / GND / SDA / SCL) |
| **Magnet** | 3 × 2 mm N52 diametric, mounted in finger CAD |
| **Quantity per glove** | 4 (Index MCP + PIP, three-finger MCP + PIP) |

## Files

| Path | What |
|---|---|
| `joint_sensor.kicad_pro` / `.kicad_pcb` / `.kicad_sch` | KiCad 7 project |
| `joint_sensor.step` | 3D model |
| `mfr/joint_sensor.zip` | Gerber + drill bundle |
| `mfr/joint_sensor-all-pos.csv` | Component positions for SMT |

## What it produces

Three int16 readings per sensor (Bx, By, Bz). The host derives:

- **Angle** from `atan2(By, Bx)` — continuous through wrap, no discontinuity.
- **Force** from `Bz` magnitude — the axial-slit PIP flexure deflects under load, changing the magnet's standoff and shifting Bz. Calibrated against a precision scale: **+1.29 counts/g, R² = 0.64** on the compliant 3F PIP joint (range ~300–1300 g).

The policy receives raw `(Bx, By, Bz)` per sensor — no on-host atan2 — so the network sees angle and force as a single 12-D continuous representation.

## Layout rules

1. **Sensor on the centerline**, aligned with the hinge-pin axis in CAD.
2. **No copper under the chip** — solid pours induce eddy currents under a moving magnet that bias the field reading. KiCad keep-out zone, ~5 × 5 mm, both layers.
3. **15 mm lever arm** between the sensor center and the M2 mounting holes. The flexure deflection that gives Bz its force-sensing range scales with this distance.

## License

MIT — see repository root.
