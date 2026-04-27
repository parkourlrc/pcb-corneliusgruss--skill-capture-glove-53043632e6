# DAQ PCB

Sensor aggregation and timing backbone for the glove. Streams synchronized proprioceptive data over USB-CDC to the host.

| | |
|---|---|
| **MCU** | STM32G431CBT6 (LQFP-48, 170 MHz, crystal-less USB via CRS) |
| **IMU** | QMI8658C — 6-DoF, I²C @ 0x6A |
| **I²C mux** | TCA9548A @ 0x70 — 6 sensor channels |
| **Sensors (off-board)** | 4× MLX90393 Hall (joint), 2× VL53L1X ToF |
| **Camera sync** | TIM1 hardware PWM, 30 Hz, 3.3 V on JST-SH |
| **Host link** | USB-C, CDC-ACM @ 12 Mbps |
| **Stack-up** | 2-layer FR4, 1.6 mm, HASL |
| **Form factor** | ~40 × 30 mm |
| **Fabricator** | JLCPCB (PCB + economic SMT assembly) |

## Files

| Path | What |
|---|---|
| `glove_daq.kicad_pro` / `.kicad_pcb` / `.kicad_sch` | KiCad 7 project |
| `glove_daq.pdf` | Schematic export |
| `glove_daq.step` | 3D model (board + components) |
| `BOM.xlsx` | Bill of materials |
| `JLCPCB_BOM.csv` | JLC-formatted BOM for SMT assembly |
| `mfr/glove_daq.zip` | Gerber + drill bundle (drop into JLCPCB upload) |
| `mfr/glove_daq-all-pos.csv` | Component positions for SMT assembly |
| `aruco/` | ArUco marker footprints + reference renders |

## Manufacturing

Upload `mfr/glove_daq.zip` (gerbers + drill) and `mfr/glove_daq-all-pos.csv` (positions) plus `JLCPCB_BOM.csv` to JLCPCB's PCBA flow. SWD header (J2, 1×4 2.54 mm THT) is hand-soldered.

## I²C mux assignment

| Channel | Sensor | Address |
|---|---|---|
| CH0 | VL53L1X ToF (bottom) | 0x29 |
| CH1 | MLX90393 Hall (3F MCP) | 0x0C |
| CH2 | MLX90393 Hall (Index PIP) | 0x0C |
| CH3 | MLX90393 Hall (3F PIP) | 0x0C |
| CH4 | MLX90393 Hall (Index MCP) | 0x0C |
| CH5 | VL53L1X ToF (top) | 0x29 |

## USB packet format (40 bytes, type 0x02 @ 30 Hz)

| Offset | Size | Field |
|---|---|---|
| 0 | 1 | Packet type (0x02) |
| 1 | 4 | Timestamp (ns) |
| 5 | 24 | 4× Hall (Bx, By, Bz int16) |
| 29 | 4 | 2× ToF (mm, int16; −1 = no reading) |
| 33 | 4 | Sync pulse timestamp (µs of last camera trigger) |
| 37 | 1 | Status flags |
| 38 | 2 | Sync sequence counter |

A separate type-0x01 packet streams raw IMU at ~680 Hz.

## Design notes

- **Crystal-less USB.** STM32G4's Clock Recovery System trims HSI48 against USB SOF — eliminates the external crystal, removes a failure mode, frees board area. Init order matters: enable HSI48 → enable CRS → select HSI48 as USB clock → init USB peripheral.
- **No battery, no SD card.** Tethered operation is required for video offload anyway, so battery + power-path management would add ~10 components for no functional gain. SPI2 (PB12–PB15) is left free for future logging.
- **JST-SH (1.0 mm) connectors.** Same physical footprint as Qwiic / STEMMA QT, half the area of JST-PH. SMD right-angle variants keep the board flat for IMU rigidity.
- **IMU substitution.** JLCPCB's "ICM-42688-PC" extended part (C2839811) is a pin-compatible **clone** — the actual silicon is QMI8658C, with a different register map and I²C-only operation. Firmware bit-bangs I²C on the SPI1 pads (PA5/SCK→SCL, PA7/MOSI→SDA, PA4/CS held HIGH for I²C mode). Lesson: never use JLC "extended" parts for specialty ICs without verifying WHO_AM_I.
- **Bit-bang I²C everywhere.** STM32 HAL's I²C peripheral hung intermittently under sensor load (mux switching + clock stretching). Bit-banging is slower but deterministic.

## License

MIT — see repository root.
