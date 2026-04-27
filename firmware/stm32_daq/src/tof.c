/**
 * tof.c — VL53L1X Time-of-Flight distance sensor driver (bit-bang I2C)
 *
 * Based on ST VL53L1X ULD (Ultra Lite Driver) API.
 * Uses 16-bit register addresses via i2c_bb_read/write_reg16.
 */

#include "tof.h"
#include "config.h"
#include "i2c_mux.h"
#include "usb_cdc.h"
#include "stm32g4xx_hal.h"

#define VL53L1X_ADDR7                   0x29

/* Register addresses */
#define SOFT_RESET                      0x0000
#define VHV_CONFIG__TIMEOUT_MACROP_LOOP_BOUND 0x0008
#define SYSTEM__INTERRUPT_CLEAR         0x0086
#define SYSTEM__MODE_START              0x0087
#define GPIO__TIO_HV_STATUS             0x0031
#define GPIO__HV_MUX_CTRL               0x0030
#define RESULT__RANGE_STATUS            0x0089
#define RESULT__FINAL_RANGE_MM_SD0      0x0096
#define FIRMWARE__SYSTEM_STATUS         0x00E5
#define IDENTIFICATION__MODEL_ID        0x010F

/* ST ULD default configuration (registers 0x2D-0x87, 91 bytes) */
static const uint8_t VL53L1X_DEFAULT_CONFIG[] = {
    0x00, /* 0x2d */ 0x00, /* 0x2e */ 0x00, /* 0x2f */
    0x01, /* 0x30 */ 0x02, /* 0x31 */ 0x00, /* 0x32 */
    0x02, /* 0x33 */ 0x08, /* 0x34 */ 0x00, /* 0x35 */
    0x08, /* 0x36 */ 0x10, /* 0x37 */ 0x01, /* 0x38 */
    0x01, /* 0x39 */ 0x00, /* 0x3a */ 0x00, /* 0x3b */
    0x00, /* 0x3c */ 0x00, /* 0x3d */ 0xff, /* 0x3e */
    0x00, /* 0x3f */ 0x0F, /* 0x40 */ 0x00, /* 0x41 */
    0x00, /* 0x42 */ 0x00, /* 0x43 */ 0x00, /* 0x44 */
    0x00, /* 0x45 */ 0x20, /* 0x46 */ 0x0b, /* 0x47 */
    0x00, /* 0x48 */ 0x00, /* 0x49 */ 0x02, /* 0x4a */
    0x0a, /* 0x4b */ 0x21, /* 0x4c */ 0x00, /* 0x4d */
    0x00, /* 0x4e */ 0x05, /* 0x4f */ 0x00, /* 0x50 */
    0x00, /* 0x51 */ 0x00, /* 0x52 */ 0x00, /* 0x53 */
    0xc8, /* 0x54 */ 0x00, /* 0x55 */ 0x00, /* 0x56 */
    0x38, /* 0x57 */ 0xff, /* 0x58 */ 0x01, /* 0x59 */
    0x00, /* 0x5a */ 0x08, /* 0x5b */ 0x00, /* 0x5c */
    0x00, /* 0x5d */ 0x01, /* 0x5e */ 0xcc, /* 0x5f */
    0x0f, /* 0x60 */ 0x01, /* 0x61 */ 0xf1, /* 0x62 */
    0x0d, /* 0x63 */ 0x01, /* 0x64 */ 0x68, /* 0x65 */
    0x00, /* 0x66 */ 0x80, /* 0x67 */ 0x08, /* 0x68 */
    0xb8, /* 0x69 */ 0x00, /* 0x6a */ 0x00, /* 0x6b */
    0x00, /* 0x6c */ 0x00, /* 0x6d */ 0x0f, /* 0x6e */
    0x89, /* 0x6f */ 0x00, /* 0x70 */ 0x00, /* 0x71 */
    0x00, /* 0x72 */ 0x00, /* 0x73 */ 0x00, /* 0x74 */
    0x00, /* 0x75 */ 0x00, /* 0x76 */ 0x01, /* 0x77 */
    0x0f, /* 0x78 */ 0x0d, /* 0x79 */ 0x0e, /* 0x7a */
    0x0e, /* 0x7b */ 0x00, /* 0x7c */ 0x00, /* 0x7d */
    0x02, /* 0x7e */ 0xc7, /* 0x7f */ 0xff, /* 0x80 */
    0x9B, /* 0x81 */ 0x00, /* 0x82 */ 0x00, /* 0x83 */
    0x00, /* 0x84 */ 0x01, /* 0x85 */ 0x00, /* 0x86 */
    0x00, /* 0x87 */
};

/* ── Register I/O ────────────────────────────────────────────────────── */

static bool wr8(uint16_t reg, uint8_t val)
{
    return i2c_bb_write_reg16(VL53L1X_ADDR7, reg, &val, 1);
}

static bool rd8(uint16_t reg, uint8_t *val)
{
    return i2c_bb_read_reg16(VL53L1X_ADDR7, reg, val, 1);
}

static bool rd16(uint16_t reg, uint16_t *val)
{
    uint8_t d[2];
    if (!i2c_bb_read_reg16(VL53L1X_ADDR7, reg, d, 2))
        return false;
    *val = ((uint16_t)d[0] << 8) | d[1];
    return true;
}

/* ── Helpers ─────────────────────────────────────────────────────────── */

static bool wait_boot(void)
{
    uint8_t s = 0;
    for (int i = 0; i < 200; i++) {
        if (rd8(FIRMWARE__SYSTEM_STATUS, &s) && (s & 0x01))
            return true;
        usb_cdc_task();  /* Keep USB alive during blocking wait */
        HAL_Delay(2);
    }
    return false;
}

static bool check_data_ready(void)
{
    uint8_t mux, status;
    if (!rd8(GPIO__HV_MUX_CTRL, &mux)) return false;
    uint8_t int_pol = !((mux & 0x10) >> 4);  /* ST ULD: inverted */
    if (!rd8(GPIO__TIO_HV_STATUS, &status)) return false;
    return (status & 0x01) == int_pol;
}

/* ── State tracking ──────────────────────────────────────────────────── */

static uint8_t fail_count[NUM_TOF_SENSORS] = {0};
static uint16_t last_good[NUM_TOF_SENSORS] = {0xFFFF, 0xFFFF};

#define TOF_MAX_READY_RETRIES   5    /* Attempts to poll data-ready per read */
#define TOF_RETRY_DELAY_US      500  /* Microseconds between ready polls */
#define TOF_FAIL_REINIT_THRESH  10   /* Consecutive failures before re-init */

/* ── Public API ───────────────────────────────────────────────────────── */

bool tof_init(uint8_t mux_channel)
{
    if (!i2c_mux_select(mux_channel))
        return false;

    /* Wait for boot */
    if (!wait_boot())
        return false;

    /* Verify model ID */
    uint8_t id;
    if (!rd8(IDENTIFICATION__MODEL_ID, &id) || id != 0xEA)
        return false;

    /* Soft reset */
    wr8(SOFT_RESET, 0x00);
    HAL_Delay(10);
    wr8(SOFT_RESET, 0x01);
    HAL_Delay(10);
    if (!wait_boot())
        return false;

    /* Write ST default config (0x2D-0x87) */
    for (int i = 0; i < 91; i++) {
        if (!wr8(0x2D + i, VL53L1X_DEFAULT_CONFIG[i]))
            return false;
    }

    /* ST ULD SensorInit: start → wait data ready → clear → stop → VHV config */
    wr8(SYSTEM__MODE_START, 0x40);  /* start ranging */
    for (int i = 0; i < 200; i++) {
        if (check_data_ready()) break;
        usb_cdc_task();
        HAL_Delay(5);
    }
    wr8(SYSTEM__INTERRUPT_CLEAR, 0x01);
    wr8(SYSTEM__MODE_START, 0x00);  /* stop ranging */
    wr8(VHV_CONFIG__TIMEOUT_MACROP_LOOP_BOUND, 0x09);
    wr8(0x0B, 0x00);

    return true;
}

bool tof_start_ranging(uint8_t mux_channel)
{
    if (!i2c_mux_select(mux_channel))
        return false;
    wr8(SYSTEM__INTERRUPT_CLEAR, 0x01);
    return wr8(SYSTEM__MODE_START, 0x40);
}

static uint16_t tof_read_single(uint8_t mux_channel, uint8_t sensor_idx)
{
    if (!i2c_mux_select(mux_channel))
        return 0xFFFF;

    /* Retry data-ready check a few times instead of single-shot */
    bool ready = false;
    for (int i = 0; i < TOF_MAX_READY_RETRIES; i++) {
        if (check_data_ready()) {
            ready = true;
            break;
        }
        /* Brief spin-wait — short enough to not block main loop significantly */
        for (volatile int d = 0; d < 300; d++) {}
    }

    if (!ready) {
        fail_count[sensor_idx]++;
        return 0xFFFF;
    }

    uint16_t distance;
    if (!rd16(RESULT__FINAL_RANGE_MM_SD0, &distance)) {
        fail_count[sensor_idx]++;
        return 0xFFFF;
    }

    /* Clear interrupt so sensor starts next measurement in continuous mode */
    if (!wr8(SYSTEM__INTERRUPT_CLEAR, 0x01)) {
        /* If clear fails, sensor won't start next ranging cycle.
         * Force-restart ranging to recover. */
        wr8(SYSTEM__MODE_START, 0x00);
        HAL_Delay(1);
        wr8(SYSTEM__INTERRUPT_CLEAR, 0x01);
        wr8(SYSTEM__MODE_START, 0x40);
    }

    if (distance > 4000) {
        fail_count[sensor_idx]++;
        return 0xFFFF;
    }

    /* Success — reset failure counter, cache value */
    fail_count[sensor_idx] = 0;
    last_good[sensor_idx] = distance;
    return distance;
}

uint16_t tof_read_distance(uint8_t mux_channel)
{
    uint8_t idx = (mux_channel == MUX_CH_TOF_TOP) ? 0 : 1;
    uint16_t val = tof_read_single(mux_channel, idx);

    /* Auto-recover: re-init + restart if too many consecutive failures */
    if (val == 0xFFFF && fail_count[idx] >= TOF_FAIL_REINIT_THRESH) {
        if (tof_init(mux_channel)) {
            tof_start_ranging(mux_channel);
            fail_count[idx] = 0;
        }
    }

    return val;
}

void tof_read_all(uint16_t distances[2])
{
    distances[0] = tof_read_distance(MUX_CH_TOF_TOP);
    distances[1] = tof_read_distance(MUX_CH_TOF_BOT);
}

uint8_t tof_get_fail_count(uint8_t sensor_idx)
{
    if (sensor_idx >= NUM_TOF_SENSORS) return 0;
    return fail_count[sensor_idx];
}

uint16_t tof_get_last_good(uint8_t sensor_idx)
{
    if (sensor_idx >= NUM_TOF_SENSORS) return 0xFFFF;
    return last_good[sensor_idx];
}
