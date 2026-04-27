/**
 * main.c — Glove DAQ firmware entry point
 *
 * Two timing domains:
 *   1. IMU at 1kHz via polled bit-bang I2C (QMI8658C on SPI pins)
 *   2. Proprioceptive loop at 100Hz (cooperative main loop)
 *
 * Both produce USB packets on the same CDC stream:
 *   - Type 0x01: IMU frames (from ring buffer)
 *   - Type 0x02: Proprioceptive frames (packed each iteration)
 */

#include "stm32g4xx_hal.h"
#include "config.h"
#include "usb_cdc.h"
#include "imu.h"
#include "i2c_mux.h"
#include "hall_sensor.h"
#include "tof.h"
#include "sync.h"
#include "led.h"
#include "button.h"
#include "frame.h"
#include "tusb.h"
#include <string.h>

/* Defined in system_clock.c */
extern void system_clock_init(void);
extern void timestamp_init(void);
extern void timestamp_update(void);
extern uint32_t timestamp_us(void);

/* ── Sensor initialization status ─────────────────────────────────────── */
static bool imu_ok = false;
static bool mux_ok = false;
static bool tof_ok[NUM_TOF_SENSORS] = {false};
static bool hall_ok[NUM_HALL_SENSORS] = {false};
static bool recording = false;

/* ── Boot LED pattern: blink SYS LED to show init progress ────────────── */

static void boot_blink(int count)
{
    for (int i = 0; i < count; i++) {
        led_sys_toggle();
        HAL_Delay(100);
        led_sys_toggle();
        HAL_Delay(100);
    }
}

/* ── Boot diagnostics over USB CDC ───────────────────────────────────── */

static bool bus_ok = false;

static void diag_puts(const char *s)
{
    usb_cdc_write((const uint8_t *)s, strlen(s));
}

static void diag_hex(uint8_t val)
{
    char buf[5];
    static const char hex[] = "0123456789abcdef";
    buf[0] = '0'; buf[1] = 'x';
    buf[2] = hex[(val >> 4) & 0xF];
    buf[3] = hex[val & 0xF];
    buf[4] = '\0';
    diag_puts(buf);
}

static void diag_int(int val)
{
    char buf[12];
    int len = 0;
    if (val == 0) { buf[0] = '0'; len = 1; }
    else {
        if (val < 0) { buf[len++] = '-'; val = -val; }
        char tmp[10]; int tl = 0;
        while (val > 0) { tmp[tl++] = '0' + (val % 10); val /= 10; }
        for (int i = tl - 1; i >= 0; i--) buf[len++] = tmp[i];
    }
    buf[len] = '\0';
    diag_puts(buf);
}

static void send_diag(void)
{
    diag_puts("\r\n=== Glove DAQ v2.0 Boot Diag ===\r\n");

    /* I2C bus scan (bit-bang — HAL I2C doesn't work on this board) */
    diag_puts("I2C bus: ");
    diag_puts(bus_ok ? "OK" : "FAIL");
    diag_puts("\r\n");

    if (bus_ok) {
        uint8_t addrs[16];
        uint8_t count = i2c_scan(addrs, 16);
        diag_puts("Bus scan: ");
        diag_int(count);
        diag_puts(" devices\r\n");
        for (uint8_t i = 0; i < count; i++) {
            diag_puts("  ");
            diag_hex(addrs[i]);
            if (addrs[i] >= 0x70 && addrs[i] <= 0x77) diag_puts(" TCA9548A mux");
            else if (addrs[i] == 0x29) diag_puts(" VL53L1X ToF");
            else if (addrs[i] == 0x0C) diag_puts(" MLX90393 Hall");
            diag_puts("\r\n");
        }
    }

    /* Mux channel scan */
    if (mux_ok) {
        diag_puts("Mux channels:\r\n");
        static const char *ch_names[] = {
            "0 tof_bot",          "1 hall_3f_mcp",
            "2 hall_idx_pip",     "3 hall_3f_pip",
            "4 hall_idx_mcp",     "5 tof_top",
            "6 (n/c)",            "7 (n/c)",
        };
        for (uint8_t ch = 0; ch < 8; ch++) {
            if (!i2c_mux_select(ch)) {
                diag_puts("  "); diag_puts(ch_names[ch]); diag_puts(": select FAIL\r\n");
                continue;
            }
            uint8_t ch_addrs[8];
            uint8_t ch_count = i2c_scan(ch_addrs, 8);
            diag_puts("  "); diag_puts(ch_names[ch]); diag_puts(": ");
            if (ch_count == 0) {
                diag_puts("empty");
            } else {
                for (uint8_t j = 0; j < ch_count; j++) {
                    if (j > 0) diag_puts(", ");
                    diag_hex(ch_addrs[j]);
                }
            }
            diag_puts("\r\n");
        }
        i2c_mux_disable();
    }

    diag_puts("IMU: "); diag_puts(imu_ok ? "OK (QMI8658C)" : "FAIL"); diag_puts("\r\n");
    diag_puts("Mux: "); diag_puts(mux_ok ? "OK (TCA9548A)" : "NOT FOUND"); diag_puts("\r\n");
    for (int i = 0; i < NUM_HALL_SENSORS; i++) {
        diag_puts("Hall["); diag_int(i); diag_puts("]: ");
        if (hall_ok[i]) {
            uint8_t g = 0xFF;
            bool got = hall_sensor_get_gain(hall_sensor_channel(i), &g);
            diag_puts("OK (MLX90393, GAIN_SEL=");
            if (got) diag_int(g); else diag_puts("?");
            diag_puts(")");
        } else {
            diag_puts("not found");
        }
        diag_puts("\r\n");
    }
    diag_puts("ToF[0] top: "); diag_puts(tof_ok[0] ? "OK" : "not found"); diag_puts("\r\n");
    diag_puts("ToF[1] bot: "); diag_puts(tof_ok[1] ? "OK" : "not found"); diag_puts("\r\n");
    diag_puts("=== END DIAG ===\r\n");

    usb_cdc_flush();
}

/* ── Initialization sequence ──────────────────────────────────────────── */

static void init_all(void)
{
    /* Core */
    system_clock_init();
    timestamp_init();

    /* UI */
    led_init();
    button_init();
    led_sys_set(true);  /* SYS LED on during init */

    /* USB CDC */
    usb_cdc_init();
    boot_blink(1);

    /* I2C bus init (peripheral only — separate from mux probe) */
    bus_ok = i2c_bus_init();
    boot_blink(1);

    /* Probe TCA9548A mux */
    if (bus_ok)
        mux_ok = i2c_mux_probe();

    if (mux_ok) {
        /* Hall sensors. NVRAM default gain saturated three of four sensors —
         * configure every chip to MLX90393_GAIN_SEL after init. */
        for (int i = 0; i < NUM_HALL_SENSORS; i++) {
            uint8_t ch = hall_sensor_channel(i);
            hall_ok[i] = hall_sensor_init(ch);
            if (hall_ok[i]) {
                hall_ok[i] = hall_sensor_configure_gain(ch, MLX90393_GAIN_SEL);
            }
        }

        /* ToF sensors */
        tof_ok[0] = tof_init(MUX_CH_TOF_TOP);   /* J8 */
        tof_ok[1] = tof_init(MUX_CH_TOF_BOT);   /* J9 */

        if (tof_ok[0]) tof_start_ranging(MUX_CH_TOF_TOP);
        if (tof_ok[1]) tof_start_ranging(MUX_CH_TOF_BOT);

        boot_blink(2);
    }

    /* IMU — QMI8658C via bit-bang I2C on SPI pins */
    imu_ok = imu_init();
    boot_blink(1);

    /* Camera sync — 30Hz PWM on PA8 */
    sync_init();
    sync_start();

    /* Boot diag is now on-demand via 'S' command over CDC */

    /* Init complete */
    led_sys_set(false);
    HAL_Delay(200);
    led_sys_set(true);  /* Steady SYS LED = running */
}

/* ── Main loop ────────────────────────────────────────────────────────── */

int main(void)
{
    init_all();

    uint32_t last_proprio_us = timestamp_us();
    uint32_t proprio_count = 0;
    uint8_t pkt_buf[PKT_SIZE_PROPRIO];
    uint16_t tof_cached[2] = {0xFFFF, 0xFFFF};
    bool hall_measurement_pending = false;
    hall_data_t hall_cached[HALL_DATA_SLOTS];
    memset(hall_cached, 0, sizeof(hall_cached));

    while (1) {
        /* ── Service USB stack ────────────────────────────────────────── */
        usb_cdc_task();

        /* ── Poll IMU ─────────────────────────────────────────────── */
        if (imu_ok)
            imu_poll();

        /* ── Service USB again after IMU poll ─────────────────────────── */
        usb_cdc_task();

        /* ── Drain IMU ring buffer → USB ──────────────────────────────── */
        {
            imu_frame_t imu;
            uint8_t imu_buf[PKT_SIZE_IMU];
            while (imu_available()) {
                imu_read(&imu);
                frame_pack_imu(&imu, imu_buf);
                usb_cdc_write(imu_buf, PKT_SIZE_IMU);
            }
        }

        /* ── Proprioceptive loop at 100Hz ─────────────────────────────── */
        uint32_t now = timestamp_us();
        if ((now - last_proprio_us) >= PROPRIO_PERIOD_US) {
            last_proprio_us = now;
            proprio_count++;

            /* Read Hall results from previous cycle's SM commands.
             * At 30Hz proprio, each cycle aligns with a camera frame. */
            if (mux_ok && hall_measurement_pending) {
                for (int i = 0; i < NUM_HALL_SENSORS; i++)
                    hall_sensor_read(hall_sensor_channel(i), &hall_cached[i]);
                hall_measurement_pending = false;
            }

            /* Read ToF (also at 30Hz — matches proprio rate now) */
            if (tof_ok[0]) {
                uint16_t d = tof_read_distance(MUX_CH_TOF_TOP);
                if (d != 0xFFFF) tof_cached[0] = d;
            }
            if (tof_ok[1]) {
                uint16_t d = tof_read_distance(MUX_CH_TOF_BOT);
                if (d != 0xFFFF) tof_cached[1] = d;
            }

            /* Get sync pulse timestamp and sequence counter */
            uint32_t sync_us = sync_get_timestamp();
            uint16_t sync_seq = sync_get_count();

            /* Build status flags */
            uint8_t status = 0;
            if (recording)
                status |= STATUS_RECORDING;
            if (sync_us != 0)
                status |= STATUS_SYNC_FIRED;
            if (button_is_pressed())
                status |= STATUS_BTN_PRESSED;

            /* Pack and send (always uses cached Hall values) */
            frame_pack_proprio(now, hall_cached, tof_cached, sync_us, status, sync_seq, pkt_buf);
            usb_cdc_write(pkt_buf, PKT_SIZE_PROPRIO);

            /* Start next Hall measurement (results read next proprio cycle, ~33ms later) */
            if (mux_ok) {
                for (int i = 0; i < NUM_HALL_SENSORS; i++)
                    hall_sensor_start_measurement(hall_sensor_channel(i));
                hall_measurement_pending = true;
            }
        }

        /* ── Host commands over CDC ───────────────────────────────────── */
        if (tud_cdc_available()) {
            uint8_t cmd;
            tud_cdc_read(&cmd, 1);
            if (cmd == 'S' || cmd == 's') {
                send_diag();
            } else if (cmd == 'T' || cmd == 't') {
                /* ToF diagnostic */
                diag_puts("\r\n=== ToF Status ===\r\n");
                diag_puts("ToF[0] init: "); diag_puts(tof_ok[0] ? "OK" : "FAIL");
                diag_puts("  fails: "); diag_int(tof_get_fail_count(0));
                diag_puts("  last_good: "); diag_int(tof_get_last_good(0));
                diag_puts(" mm\r\n");
                diag_puts("ToF[1] init: "); diag_puts(tof_ok[1] ? "OK" : "FAIL");
                diag_puts("  fails: "); diag_int(tof_get_fail_count(1));
                diag_puts("  last_good: "); diag_int(tof_get_last_good(1));
                diag_puts(" mm\r\n");
                /* Try a live read */
                if (tof_ok[0]) {
                    uint16_t d = tof_read_distance(MUX_CH_TOF_TOP);
                    diag_puts("ToF[0] live: ");
                    if (d == 0xFFFF) diag_puts("NO DATA");
                    else { diag_int(d); diag_puts(" mm"); }
                    diag_puts("\r\n");
                }
                if (tof_ok[1]) {
                    uint16_t d = tof_read_distance(MUX_CH_TOF_BOT);
                    diag_puts("ToF[1] live: ");
                    if (d == 0xFFFF) diag_puts("NO DATA");
                    else { diag_int(d); diag_puts(" mm"); }
                    diag_puts("\r\n");
                }
                diag_puts("=== END ToF ===\r\n");
                usb_cdc_flush();
            } else if (cmd == 'R' || cmd == 'r') {
                /* Host says recording started — turn on REC LED */
                recording = true;
                led_rec_set(true);
            } else if (cmd == 'O' || cmd == 'o') {
                /* Host says recording stopped — turn off REC LED */
                recording = false;
                led_rec_set(false);
            } else if (cmd == 'H' || cmd == 'h') {
                /* Hall sensor diagnostic — scan all mux channels for MLX90393 */
                diag_puts("\r\n=== Hall Sensor Diag ===\r\n");

                /* Show init status */
                for (int i = 0; i < NUM_HALL_SENSORS; i++) {
                    diag_puts("Hall["); diag_int(i); diag_puts("] init: ");
                    diag_puts(hall_ok[i] ? "OK" : "FAIL");
                    diag_puts("\r\n");
                }

                /* Scan all 8 mux channels for 0x0C */
                diag_puts("MLX90393 scan (addr 0x0C):\r\n");
                for (uint8_t ch = 0; ch < 8; ch++) {
                    if (!i2c_mux_select(ch)) continue;
                    uint8_t addrs[8];
                    uint8_t cnt = i2c_scan(addrs, 8);
                    for (uint8_t j = 0; j < cnt; j++) {
                        if (addrs[j] == 0x0C) {
                            diag_puts("  CH"); diag_int(ch);
                            diag_puts(": MLX90393 found!\r\n");

                            /* Try init + single read on this channel */
                            if (hall_sensor_init(ch)) {
                                diag_puts("    init: OK\r\n");
                                if (hall_sensor_start_measurement(ch)) {
                                    HAL_Delay(2);
                                    hall_data_t hd;
                                    if (hall_sensor_read(ch, &hd)) {
                                        diag_puts("    Bx="); diag_int(hd.bx);
                                        diag_puts(" By="); diag_int(hd.by);
                                        diag_puts(" Bz="); diag_int(hd.bz);
                                        diag_puts("\r\n");
                                        /* Take a few more readings */
                                        for (int r = 0; r < 4; r++) {
                                            hall_sensor_start_measurement(ch);
                                            HAL_Delay(2);
                                            if (hall_sensor_read(ch, &hd)) {
                                                diag_puts("    Bx="); diag_int(hd.bx);
                                                diag_puts(" By="); diag_int(hd.by);
                                                diag_puts(" Bz="); diag_int(hd.bz);
                                                diag_puts("\r\n");
                                            }
                                        }
                                    } else {
                                        diag_puts("    read FAIL\r\n");
                                    }
                                } else {
                                    diag_puts("    SM FAIL\r\n");
                                }
                            } else {
                                diag_puts("    init FAIL\r\n");
                            }
                        }
                    }
                }
                i2c_mux_disable();

                diag_puts("=== END Hall ===\r\n");
                usb_cdc_flush();
            } else if (cmd == 'G' || cmd == 'g') {
                /* Hall GAIN_SEL readback — non-destructive (no reset/init) */
                diag_puts("\r\n=== Hall GAIN_SEL ===\r\n");
                for (int i = 0; i < NUM_HALL_SENSORS; i++) {
                    uint8_t g = 0xFF;
                    bool ok = hall_sensor_get_gain(hall_sensor_channel(i), &g);
                    diag_puts("Hall["); diag_int(i); diag_puts("] GAIN_SEL=");
                    if (ok) diag_int(g); else diag_puts("READ_FAIL");
                    diag_puts("\r\n");
                }
                i2c_mux_disable();
                diag_puts("=== END GAIN ===\r\n");
                usb_cdc_flush();
            }
        }

        /* ── Button handling ──────────────────────────────────────────── */
        button_update();
        if (button_was_pressed()) {
            recording = !recording;
            led_rec_set(recording);
        }

        /* ── Flush USB TX buffer ──────────────────────────────────────── */
        usb_cdc_flush();
    }
}
