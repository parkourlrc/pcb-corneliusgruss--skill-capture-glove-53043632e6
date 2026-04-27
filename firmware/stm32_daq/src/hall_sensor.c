/**
 * hall_sensor.c — MLX90393 3D Hall effect sensor driver (bit-bang I2C)
 *
 * The MLX90393 uses a command-based I2C protocol (not register-based).
 * Each transaction uses a repeated start between write and read:
 *   START → addr+W → cmd [→ data] → Sr → addr+R → status [→ data] → STOP
 *
 * Commands:
 *   SM (0x3E): Start Measurement X+Y+Z+T
 *   RM (0x4E): Read Measurement → status + T(2) + X(2) + Y(2) + Z(2)
 *   WR (0x60): Write Register → [cmd, data_hi, data_lo, reg<<2] → status
 *   RR (0x50): Read Register → [cmd, reg<<2] → status + data(2)
 *   RT (0xF0): Reset → status
 *   EX (0x80): Exit mode → status
 *
 * Status byte: bit 3 = ERROR, bit 4 = SED, bit 5 = RS (see MLX_STATUS_ERROR)
 *
 * Data is raw Bx/By/Bz as int16 — angle and force decomposition happens
 * on the host in Python (hall_calibration.py).
 */

#include "hall_sensor.h"
#include "config.h"
#include "i2c_mux.h"
#include "stm32g4xx_hal.h"
#include <string.h>

#define MLX_ADDR7  0x0C  /* 7-bit I2C address (A0=A1=GND) */

/* Mux channels for 4 Hall sensors — slot order matches packet layout:
 * [0] index_mcp, [1] index_pip, [2] 3f_mcp, [3] 3f_pip */
static const uint8_t hall_channels[NUM_HALL_SENSORS] = {
    MUX_CH_HALL_INDEX_MCP,
    MUX_CH_HALL_INDEX_PIP,
    MUX_CH_HALL_3F_MCP,
    MUX_CH_HALL_3F_PIP,
};

/* ── Low-level MLX90393 I2C command helpers (bit-bang) ───────────────── */

/*
 * MLX90393 status byte (returned after every command):
 *   Bit 0: BURST mode active
 *   Bit 1: WOC mode active
 *   Bit 2: SM mode active
 *   Bit 3: ERROR
 *   Bit 4: SED (single error in EEPROM)
 *   Bit 5: RS (reset occurred)
 *   Bits 7:6: response byte count - 1
 */
#define MLX_STATUS_ERROR  (1 << 3)

/**
 * Send command and read status in a single I2C transaction (repeated start).
 * MLX90393 requires: START → W[cmd] → Sr → R[status] → STOP
 */
static bool mlx_send_cmd(uint8_t cmd, uint8_t *status_out)
{
    uint8_t st;
    if (!i2c_bb_write_read(MLX_ADDR7, &cmd, 1, &st, 1))
        return false;

    if (status_out)
        *status_out = st;

    return !(st & MLX_STATUS_ERROR);
}

/**
 * Write a 16-bit value to an MLX90393 register.
 * WR format: [0x60, data_hi, data_lo, reg_addr<<2] → status
 */
static bool mlx_write_reg(uint8_t reg_addr, uint16_t value)
{
    uint8_t tx[4] = {
        MLX90393_CMD_WR,
        (uint8_t)(value >> 8),
        (uint8_t)(value & 0xFF),
        (uint8_t)(reg_addr << 2),
    };

    uint8_t st;
    if (!i2c_bb_write_read(MLX_ADDR7, tx, 4, &st, 1))
        return false;

    return !(st & MLX_STATUS_ERROR);
}

/**
 * Read a 16-bit register value from MLX90393.
 * RR format: [0x50, reg_addr<<2] → status + 2 data bytes
 */
static bool mlx_read_reg(uint8_t reg_addr, uint16_t *value)
{
    uint8_t tx[2] = {MLX90393_CMD_RR, (uint8_t)(reg_addr << 2)};
    uint8_t rx[3];  /* status + data_hi + data_lo */

    if (!i2c_bb_write_read(MLX_ADDR7, tx, 2, rx, 3))
        return false;

    if (rx[0] & MLX_STATUS_ERROR)
        return false;

    *value = ((uint16_t)rx[1] << 8) | rx[2];
    return true;
}

/* ── Public API ───────────────────────────────────────────────────────── */

uint8_t hall_sensor_channel(uint8_t slot)
{
    if (slot >= NUM_HALL_SENSORS)
        return 0xFF;
    return hall_channels[slot];
}

bool hall_sensor_init(uint8_t mux_channel)
{
    if (!i2c_mux_select(mux_channel))
        return false;

    /* Reset the sensor */
    mlx_send_cmd(MLX90393_CMD_RT, NULL);
    HAL_Delay(2);

    /* Exit any active mode */
    mlx_send_cmd(MLX90393_CMD_EX, NULL);
    HAL_Delay(1);

    /* Verify communication by reading register 0 */
    uint16_t reg0;
    if (!mlx_read_reg(0x00, &reg0))
        return false;

    return true;
}

bool hall_sensor_configure_gain(uint8_t mux_channel, uint8_t gain_sel)
{
    if (gain_sel > 7)
        return false;

    if (!i2c_mux_select(mux_channel))
        return false;

    /* Read current register 0 value */
    uint16_t reg0;
    if (!mlx_read_reg(MLX90393_REG_GAIN_SEL, &reg0))
        return false;

    /* GAIN_SEL is bits [7:4] of register 0 */
    reg0 = (reg0 & ~0x00F0) | ((uint16_t)gain_sel << 4);

    if (!mlx_write_reg(MLX90393_REG_GAIN_SEL, reg0))
        return false;

    /* Verify write */
    uint16_t verify;
    if (!mlx_read_reg(MLX90393_REG_GAIN_SEL, &verify))
        return false;

    return ((verify >> 4) & 0x07) == gain_sel;
}

bool hall_sensor_get_gain(uint8_t mux_channel, uint8_t *gain_out)
{
    if (!i2c_mux_select(mux_channel))
        return false;

    uint16_t reg0;
    if (!mlx_read_reg(MLX90393_REG_GAIN_SEL, &reg0))
        return false;

    *gain_out = (uint8_t)((reg0 >> 4) & 0x07);
    return true;
}

bool hall_sensor_start_measurement(uint8_t mux_channel)
{
    if (!i2c_mux_select(mux_channel))
        return false;

    return mlx_send_cmd(MLX90393_CMD_SM_XYZ, NULL);
}

bool hall_sensor_read(uint8_t mux_channel, hall_data_t *data)
{
    if (!i2c_mux_select(mux_channel))
        return false;

    /* RM command: W[0x4E] → Sr → R[status + Bx + By + Bz]
     * Command 0x4E = RM with zyxt=1110 (X+Y+Z, no T) → 7 bytes response. */
    uint8_t cmd = MLX90393_CMD_RM_XYZ;
    uint8_t rx[7];  /* status(1) + Bx(2) + By(2) + Bz(2) */
    if (!i2c_bb_write_read(MLX_ADDR7, &cmd, 1, rx, 7))
        return false;

    if (rx[0] & MLX_STATUS_ERROR)
        return false;

    /* Extract Bx/By/Bz (big-endian, no temperature field) */
    data->bx = (int16_t)(((uint16_t)rx[1] << 8) | rx[2]);
    data->by = (int16_t)(((uint16_t)rx[3] << 8) | rx[4]);
    data->bz = (int16_t)(((uint16_t)rx[5] << 8) | rx[6]);

    return true;
}

void hall_sensor_read_all(hall_data_t data[4])
{
    /* Zero all slots — unused slots stay zero, failed reads stay zero */
    memset(data, 0, sizeof(hall_data_t) * HALL_DATA_SLOTS);

    /* Start all measurements first (minimizes total wait time) */
    int started = 0;
    for (int i = 0; i < NUM_HALL_SENSORS; i++) {
        if (hall_sensor_start_measurement(hall_channels[i]))
            started++;
    }

    if (started == 0)
        return;

    /* Wait for conversion (~1ms for XYZ+T at default OSR, 2ms for margin) */
    HAL_Delay(2);

    /* Read results — slot stays zeroed on failure */
    for (int i = 0; i < NUM_HALL_SENSORS; i++) {
        hall_sensor_read(hall_channels[i], &data[i]);
    }
}
