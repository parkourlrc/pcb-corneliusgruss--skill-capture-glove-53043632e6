/**
 * i2c_mux.c — TCA9548A I2C multiplexer driver (bit-bang I2C)
 *
 * Uses bit-bang I2C on PB6(SCL)/PB7(SDA) instead of the HAL I2C
 * peripheral. HAL I2C fails on this board for unknown timing reasons,
 * but bit-bang works reliably (same approach as QMI8658C IMU driver).
 *
 * The TCA9548A has a single control register: write a byte where bit N
 * enables channel N. We only enable one channel at a time.
 */

#include "i2c_mux.h"
#include "config.h"
#include "stm32g4xx_hal.h"

static uint8_t current_channel = 0xFF;
static bool bus_initialized = false;

/* ── Bit-bang I2C on PB6/PB7 ─────────────────────────────────────────── */

#define BB_SCL  SENSOR_I2C_SCL_PIN   /* GPIO_PIN_6 */
#define BB_SDA  SENSOR_I2C_SDA_PIN   /* GPIO_PIN_7 */
#define BB_PORT SENSOR_I2C_GPIO_PORT /* GPIOB */

static inline void bb_delay(void)
{
    /* ~3µs at 170MHz → ~166kHz I2C. Conservative for reliability. */
    for (volatile int i = 0; i < 50; i++) {}
}

static inline void scl_h(void) { BB_PORT->BSRR = BB_SCL; }
static inline void scl_l(void) { BB_PORT->BRR  = BB_SCL; }
static inline void sda_h(void) { BB_PORT->BSRR = BB_SDA; }
static inline void sda_l(void) { BB_PORT->BRR  = BB_SDA; }
static inline int  sda_r(void) { return (BB_PORT->IDR & BB_SDA) ? 1 : 0; }

static void bb_start(void)
{
    sda_h(); bb_delay();
    scl_h(); bb_delay();
    sda_l(); bb_delay();
    scl_l(); bb_delay();
}

static void bb_stop(void)
{
    sda_l(); bb_delay();
    scl_h(); bb_delay();
    sda_h(); bb_delay();
}

static int bb_write(uint8_t b)
{
    for (int i = 7; i >= 0; i--) {
        if (b & (1 << i)) sda_h(); else sda_l();
        bb_delay();
        scl_h(); bb_delay();
        scl_l(); bb_delay();
    }
    /* Read ACK */
    sda_h(); bb_delay();
    scl_h(); bb_delay();
    int nack = sda_r();
    scl_l(); bb_delay();
    return nack;  /* 0=ACK, 1=NACK */
}

static uint8_t bb_read(int send_ack)
{
    uint8_t val = 0;
    sda_h();
    for (int i = 7; i >= 0; i--) {
        bb_delay();
        scl_h(); bb_delay();
        if (sda_r()) val |= (1 << i);
        scl_l(); bb_delay();
    }
    if (send_ack) sda_l(); else sda_h();
    bb_delay();
    scl_h(); bb_delay();
    scl_l(); bb_delay();
    sda_h();
    return val;
}

/* ── Public API ──────────────────────────────────────────────────────── */

bool i2c_bus_init(void)
{
    if (bus_initialized)
        return true;

    SENSOR_I2C_GPIO_CLK_ENABLE();

    /* Configure PB6/PB7 as open-drain GPIO with internal pull-ups
     * (external 4.7k pull-ups are on the PCB) */
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = BB_SCL | BB_SDA;
    gpio.Mode = GPIO_MODE_OUTPUT_OD;
    gpio.Pull = GPIO_PULLUP;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(BB_PORT, &gpio);

    /* Release bus — both lines high */
    scl_h();
    sda_h();
    bb_delay(); bb_delay();

    bus_initialized = true;
    return true;
}

bool i2c_mux_probe(void)
{
    /* Try to write address byte to TCA9548A — ACK means present */
    bb_start();
    int nack = bb_write(MUX_I2C_ADDR);  /* Already shifted: 0x70<<1 = 0xE0 */
    bb_stop();

    if (nack)
        return false;

    return i2c_mux_disable();
}

bool i2c_mux_init(void)
{
    if (!i2c_bus_init())
        return false;
    return i2c_mux_probe();
}

bool i2c_mux_select(uint8_t channel)
{
    if (channel > 7)
        return false;

    if (channel == current_channel)
        return true;

    uint8_t mask = 1U << channel;
    bb_start();
    if (bb_write(MUX_I2C_ADDR)) { bb_stop(); return false; }
    if (bb_write(mask))          { bb_stop(); return false; }
    bb_stop();

    current_channel = channel;
    return true;
}

bool i2c_mux_disable(void)
{
    current_channel = 0xFF;
    bb_start();
    if (bb_write(MUX_I2C_ADDR)) { bb_stop(); return false; }
    if (bb_write(0x00))          { bb_stop(); return false; }
    bb_stop();
    return true;
}

uint8_t i2c_scan(uint8_t *addrs, uint8_t max_addrs)
{
    uint8_t count = 0;
    for (uint8_t addr7 = 0x08; addr7 <= 0x77 && count < max_addrs; addr7++) {
        bb_start();
        int nack = bb_write(addr7 << 1);
        bb_stop();
        if (!nack) {
            addrs[count++] = addr7;
        }
    }
    return count;
}

/* ── Bit-bang I2C read/write for downstream sensors ──────────────────── */

bool i2c_bb_write_reg(uint8_t addr7, uint8_t reg, const uint8_t *data, uint8_t len)
{
    bb_start();
    if (bb_write(addr7 << 1)) { bb_stop(); return false; }
    if (bb_write(reg))        { bb_stop(); return false; }
    for (uint8_t i = 0; i < len; i++) {
        if (bb_write(data[i])) { bb_stop(); return false; }
    }
    bb_stop();
    return true;
}

bool i2c_bb_read_reg(uint8_t addr7, uint8_t reg, uint8_t *data, uint8_t len)
{
    bb_start();
    if (bb_write(addr7 << 1))     { bb_stop(); return false; }
    if (bb_write(reg))             { bb_stop(); return false; }
    bb_start();  /* repeated start */
    if (bb_write((addr7 << 1) | 1)) { bb_stop(); return false; }
    for (uint8_t i = 0; i < len; i++) {
        data[i] = bb_read(i < len - 1);  /* ACK all but last */
    }
    bb_stop();
    return true;
}

bool i2c_bb_write_reg16(uint8_t addr7, uint16_t reg, const uint8_t *data, uint8_t len)
{
    bb_start();
    if (bb_write(addr7 << 1))           { bb_stop(); return false; }
    if (bb_write((uint8_t)(reg >> 8)))   { bb_stop(); return false; }
    if (bb_write((uint8_t)(reg & 0xFF))) { bb_stop(); return false; }
    for (uint8_t i = 0; i < len; i++) {
        if (bb_write(data[i])) { bb_stop(); return false; }
    }
    bb_stop();
    return true;
}

bool i2c_bb_read_reg16(uint8_t addr7, uint16_t reg, uint8_t *data, uint8_t len)
{
    bb_start();
    if (bb_write(addr7 << 1))           { bb_stop(); return false; }
    if (bb_write((uint8_t)(reg >> 8)))   { bb_stop(); return false; }
    if (bb_write((uint8_t)(reg & 0xFF))) { bb_stop(); return false; }
    bb_start();  /* repeated start */
    if (bb_write((addr7 << 1) | 1))      { bb_stop(); return false; }
    for (uint8_t i = 0; i < len; i++) {
        data[i] = bb_read(i < len - 1);
    }
    bb_stop();
    return true;
}

bool i2c_bb_write_raw(uint8_t addr7, const uint8_t *data, uint8_t len)
{
    bb_start();
    if (bb_write(addr7 << 1)) { bb_stop(); return false; }
    for (uint8_t i = 0; i < len; i++) {
        if (bb_write(data[i])) { bb_stop(); return false; }
    }
    bb_stop();
    return true;
}

bool i2c_bb_read_raw(uint8_t addr7, uint8_t *data, uint8_t len)
{
    bb_start();
    if (bb_write((addr7 << 1) | 1)) { bb_stop(); return false; }
    for (uint8_t i = 0; i < len; i++) {
        data[i] = bb_read(i < len - 1);  /* ACK all but last */
    }
    bb_stop();
    return true;
}

bool i2c_bb_write_read(uint8_t addr7, const uint8_t *tx, uint8_t tx_len, uint8_t *rx, uint8_t rx_len)
{
    bb_start();
    if (bb_write(addr7 << 1)) { bb_stop(); return false; }
    for (uint8_t i = 0; i < tx_len; i++) {
        if (bb_write(tx[i])) { bb_stop(); return false; }
    }
    bb_start();  /* repeated start */
    if (bb_write((addr7 << 1) | 1)) { bb_stop(); return false; }
    for (uint8_t i = 0; i < rx_len; i++) {
        rx[i] = bb_read(i < rx_len - 1);
    }
    bb_stop();
    return true;
}
