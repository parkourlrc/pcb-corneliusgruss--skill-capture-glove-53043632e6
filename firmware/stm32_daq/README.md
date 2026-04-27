# STM32 DAQ firmware

Firmware for the [DAQ PCB](../../hardware/daq_pcb/) (STM32G431CBT6). Aggregates Hall + ToF + IMU sensors and streams timestamped frames over USB-CDC at ~22 KB/s.

## Architecture

Two independent timing domains:

| Domain | Rate | Source | Sensors |
|---|---|---|---|
| IMU | ~680 Hz observed (1 kHz nominal) | EXTI → bit-bang I²C burst-read | QMI8658C (accel + gyro) |
| Proprio | 30 Hz | Free-running main loop | 4× MLX90393 Hall, 2× VL53L1X ToF |

A 30 Hz TIM1 PWM pulse on PA8 triggers the on-glove cameras. The pulse timestamp is embedded in the proprio stream so the host can match camera frames to sensor frames within <1 µs of jitter.

## Module map

```
src/
├── main.c              Init sequence + main loop
├── system_clock.c      HSI16 → PLL → 170 MHz, HSI48 + CRS for crystal-less USB
├── usb_cdc.c           TinyUSB CDC device + ring-buffer TX
├── usb_descriptors.c   USB device/config/string descriptors
├── tusb_config.h       TinyUSB compile-time configuration
├── imu.c               QMI8658C bit-bang I²C driver (placed on SPI1 pads)
├── i2c_mux.c           TCA9548A channel switching
├── hall_sensor.c       MLX90393 driver (SM/RM, gain config, raw Bx/By/Bz)
├── tof.c               VL53L1X ranging via mux
├── sync.c              30 Hz PWM camera trigger (TIM1_CH1)
├── led.c               LED_REC (PB4) + LED_SYS (PB5)
├── button.c            SW_RECORD (PB3) with debounce
├── frame.c             Sensor frame packing + timestamping
└── blink_test.c        Optional bringup blinky (BLINK_TEST=ON)
```

See [`../../hardware/daq_pcb/README.md`](../../hardware/daq_pcb/README.md) for the wire-level packet format.

## Build

External SDK dependencies live under `lib/` and are not vendored in this repo (~540 MB). Clone them once:

```bash
mkdir -p lib && cd lib
git clone https://github.com/STMicroelectronics/STM32CubeG4.git
git clone https://github.com/hathach/tinyusb.git
cd ..
```

Toolchain on macOS:

```bash
brew install arm-none-eabi-gcc cmake openocd
```

Build:

```bash
mkdir -p build && cd build
cmake -DBLINK_TEST=OFF ..
make -j8
```

Output: `build/glove_daq.elf` / `.bin`.

## Flash

Over USB DFU (no debugger needed):

```bash
dfu-util -a 0 -s 0x08000000:leave -D build/glove_daq.bin
```

Or via SWD with ST-Link:

```bash
make flash
```

## Design notes

- **Bit-bang I²C everywhere.** STM32 HAL's I²C peripheral hung intermittently under sensor load (mux switching + clock stretching from VL53L1X). Bit-banging is slower but deterministic and trivial to debug with a scope.
- **Crystal-less USB.** HSI48 is trimmed against USB SOF via the Clock Recovery System — no external crystal, fewer components, one fewer failure mode. CRS init must precede USB peripheral init or enumeration fails on HSI48 drift.
- **QMI8658C on SPI1 pads.** JLCPCB substituted a clone for the spec'd ICM-42688-P; firmware bit-bangs I²C on the SPI1 pads (PA5/SCK→SCL, PA7/MOSI→SDA, PA4/CS held HIGH for I²C mode). See [`../../hardware/daq_pcb/README.md`](../../hardware/daq_pcb/README.md) for full context.
- **Raw Hall fields, not angles.** The firmware ships `(Bx, By, Bz)` per Hall sensor untouched. Angle decomposition (atan2) and force calibration happen on the host so they can be iterated without reflashing.

## License

Application code: MIT (see repository root).

Vendor SDKs (cloned at build time, not redistributed):
- STM32CubeG4: BSD-3-Clause (STMicroelectronics)
- TinyUSB: MIT (Ha Thach)
