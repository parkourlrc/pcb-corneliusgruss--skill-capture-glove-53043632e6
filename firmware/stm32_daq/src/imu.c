/**
 * imu.c — QMI8658C driver via bit-bang I2C on SPI pins
 *
 * Hardware: QMI8658C (Tokmas ICM-42688-PC) soldered on SPI1 footprint.
 * Protocol: Bit-bang I2C on PA5(SCL) + PA7(SDA), PA4(CS) held HIGH.
 * I2C address: 0x6A (SA0/AD0 pin low).
 *
 * QMI8658C register map (QST Corporation, Rev 0.9):
 *   0x00  WHO_AM_I     = 0x05
 *   0x01  REVISION_ID
 *   0x02  CTRL1        SPI/sensor config
 *   0x03  CTRL2        Accel: [6:4]=range, [3:0]=ODR
 *   0x04  CTRL3        Gyro:  [6:4]=range, [3:0]=ODR
 *   0x06  CTRL5        Low-pass filter
 *   0x08  CTRL7        Enable: bit0=accel, bit1=gyro
 *   0x33  TEMP_L       Temperature (little-endian)
 *   0x35  AX_L         Accel data (little-endian, 6 bytes)
 *   0x3B  GX_L         Gyro data (little-endian, 6 bytes)
 *
 * Pinout difference from ICM-42688-P:
 *   Pin 4: QMI8658=INT2, ICM-42688=INT1 (PB0 on our board)
 *   Pin 9: QMI8658=INT1, ICM-42688=INT2
 */

#include "imu.h"
#include "config.h"
#include "stm32g4xx_hal.h"
/* usb_cdc.h not needed — USB task is called from main loop */

/* ── QMI8658C registers ──────────────────────────────────────────────── */
#define QMI_WHO_AM_I        0x00
#define QMI_WHO_AM_I_VAL    0x05
#define QMI_CTRL1           0x02
#define QMI_CTRL2           0x03  /* Accel config */
#define QMI_CTRL3           0x04  /* Gyro config */
#define QMI_CTRL5           0x06  /* Low-pass filter */
#define QMI_CTRL7           0x08  /* Sensor enable */
#define QMI_STATUS0         0x2E
#define QMI_TEMP_L          0x33
#define QMI_AX_L            0x35
#define QMI_GX_L            0x3B
#define QMI_I2C_ADDR        0x6A

/* CTRL2/CTRL3 ODR values (bits [3:0]) */
#define QMI_ODR_8000HZ      0x00
#define QMI_ODR_4000HZ      0x01
#define QMI_ODR_2000HZ      0x02
#define QMI_ODR_1000HZ      0x03
#define QMI_ODR_500HZ       0x04
#define QMI_ODR_250HZ       0x05

/* CTRL2 accel range (bits [6:4]) */
#define QMI_ACC_2G          (0x00 << 4)
#define QMI_ACC_4G          (0x01 << 4)
#define QMI_ACC_8G          (0x02 << 4)
#define QMI_ACC_16G         (0x03 << 4)

/* CTRL3 gyro range (bits [6:4]) */
#define QMI_GYR_16DPS       (0x00 << 4)
#define QMI_GYR_32DPS       (0x01 << 4)
#define QMI_GYR_64DPS       (0x02 << 4)
#define QMI_GYR_128DPS      (0x03 << 4)
#define QMI_GYR_256DPS      (0x04 << 4)
#define QMI_GYR_512DPS      (0x05 << 4)
#define QMI_GYR_1024DPS     (0x06 << 4)
#define QMI_GYR_2048DPS     (0x07 << 4)

/* ── Pin definitions (SPI footprint used as I2C) ─────────────────────── */
#define PIN_CS   IMU_CS_PIN      /* PA4 — held HIGH for I2C mode */
#define PIN_SCL  IMU_SCK_PIN     /* PA5 */
#define PIN_SDA  IMU_MOSI_PIN    /* PA7 */
#define PORT     IMU_SPI_GPIO_PORT  /* GPIOA */

/* ── Bit-bang I2C primitives ─────────────────────────────────────────── */

static inline void i2c_delay(void)
{
    /* ~1.2µs at 170MHz → ~800kHz I2C Fast Mode Plus */
    for (volatile int i = 0; i < 18; i++) {}
}

static inline void scl_high(void) { PORT->BSRR = PIN_SCL; }
static inline void scl_low(void)  { PORT->BRR  = PIN_SCL; }
static inline void sda_high(void) { PORT->BSRR = PIN_SDA; }
static inline void sda_low(void)  { PORT->BRR  = PIN_SDA; }
static inline int  sda_read(void) { return (PORT->IDR & PIN_SDA) ? 1 : 0; }

static void i2c_start(void)
{
    sda_high(); i2c_delay();
    scl_high(); i2c_delay();
    sda_low();  i2c_delay();
    scl_low();  i2c_delay();
}

static void i2c_stop(void)
{
    sda_low();  i2c_delay();
    scl_high(); i2c_delay();
    sda_high(); i2c_delay();
}

static int i2c_write_byte(uint8_t b)
{
    for (int i = 7; i >= 0; i--) {
        if (b & (1 << i)) sda_high(); else sda_low();
        i2c_delay();
        scl_high();
        /* Clock stretching timeout */
        for (volatile int w = 0; w < 1000; w++) {
            if (PORT->IDR & PIN_SCL) break;
        }
        i2c_delay();
        scl_low();  i2c_delay();
    }
    sda_high(); i2c_delay();
    scl_high(); i2c_delay();
    int ack = sda_read();
    scl_low();  i2c_delay();
    return ack;  /* 0=ACK, 1=NACK */
}

static uint8_t i2c_read_byte(int send_ack)
{
    uint8_t val = 0;
    sda_high();
    for (int i = 7; i >= 0; i--) {
        i2c_delay();
        scl_high(); i2c_delay();
        if (sda_read()) val |= (1 << i);
        scl_low();  i2c_delay();
    }
    if (send_ack) sda_low(); else sda_high();
    i2c_delay();
    scl_high(); i2c_delay();
    scl_low();  i2c_delay();
    sda_high();
    return val;
}

/* ── Register access ─────────────────────────────────────────────────── */

static int qmi_read_reg(uint8_t reg, uint8_t *val)
{
    i2c_start();
    if (i2c_write_byte(QMI_I2C_ADDR << 1))     { i2c_stop(); return -1; }
    if (i2c_write_byte(reg))                     { i2c_stop(); return -2; }
    i2c_start();  /* repeated start */
    if (i2c_write_byte((QMI_I2C_ADDR << 1) | 1)){ i2c_stop(); return -3; }
    *val = i2c_read_byte(0);  /* NACK last byte */
    i2c_stop();
    return 0;
}

static int qmi_read_burst(uint8_t reg, uint8_t *buf, int len)
{
    i2c_start();
    if (i2c_write_byte(QMI_I2C_ADDR << 1))     { i2c_stop(); return -1; }
    if (i2c_write_byte(reg))                     { i2c_stop(); return -2; }
    i2c_start();
    if (i2c_write_byte((QMI_I2C_ADDR << 1) | 1)){ i2c_stop(); return -3; }
    for (int i = 0; i < len; i++)
        buf[i] = i2c_read_byte(i < len - 1);  /* ACK all but last */
    i2c_stop();
    return 0;
}

static void qmi_write_reg(uint8_t reg, uint8_t val)
{
    i2c_start();
    i2c_write_byte(QMI_I2C_ADDR << 1);
    i2c_write_byte(reg);
    i2c_write_byte(val);
    i2c_stop();
}

/* ── Ring buffer ──────────────────────────────────────────────────────── */

static imu_frame_t ring[IMU_RING_SIZE];
static volatile uint32_t ring_head = 0;
static volatile uint32_t ring_tail = 0;

extern uint32_t timestamp_us(void);

/* ── Timing for polling ──────────────────────────────────────────────── */

static uint32_t last_poll_us = 0;
/* Burst read of 12 bytes ≈ 350µs per sample.
 * 1kHz poll matches IMU ODR; actual rate limited by proprio loop sharing CPU. */
#define IMU_POLL_PERIOD_US  1000U  /* 1 kHz target */

/* ── Public API ──────────────────────────────────────────────────────── */

bool imu_init(void)
{
    IMU_SPI_GPIO_CLK_ENABLE();

    /* PA4 (CS) = output push-pull, HIGH for I2C mode */
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = PIN_CS;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(PORT, &gpio);
    PORT->BSRR = PIN_CS;  /* CS HIGH → I2C mode */

    /* PA5 (SCL) + PA7 (SDA) = open-drain with pull-ups */
    gpio.Pin = PIN_SCL | PIN_SDA;
    gpio.Mode = GPIO_MODE_OUTPUT_OD;
    gpio.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(PORT, &gpio);
    scl_high();
    sda_high();

    HAL_Delay(100);  /* Wait for QMI8658C power-up */

    /* Verify WHO_AM_I */
    uint8_t who = 0;
    if (qmi_read_reg(QMI_WHO_AM_I, &who) != 0)
        return false;
    if (who != QMI_WHO_AM_I_VAL)
        return false;

    /* Enable address auto-increment for burst reads (CTRL1 bit 6) */
    qmi_write_reg(QMI_CTRL1, 0x40);

    /* Configure: Accel ±2g @ 1kHz, Gyro ±256dps @ 1kHz */
    qmi_write_reg(QMI_CTRL2, QMI_ACC_2G | QMI_ODR_1000HZ);
    qmi_write_reg(QMI_CTRL3, QMI_GYR_256DPS | QMI_ODR_1000HZ);

    /* Enable accel + gyro */
    qmi_write_reg(QMI_CTRL7, 0x03);
    HAL_Delay(50);

    /* Verify enable stuck */
    uint8_t ctrl7 = 0;
    qmi_read_reg(QMI_CTRL7, &ctrl7);
    if (ctrl7 != 0x03)
        return false;

    last_poll_us = timestamp_us();
    return true;
}

void imu_poll(void)
{
    uint32_t now = timestamp_us();
    if ((now - last_poll_us) < IMU_POLL_PERIOD_US)
        return;
    last_poll_us = now;

    /* Burst read: single I2C transaction for all 12 data bytes (0x35-0x40).
     * QMI8658C auto-increments the register pointer on sequential reads. */
    uint8_t raw[12];
    imu_frame_t frame;
    frame.timestamp_us = now;

    if (qmi_read_burst(QMI_AX_L, raw, 12) != 0)
        return;  /* I2C error — skip this sample */

    frame.accel[0] = (int16_t)(raw[1]  << 8 | raw[0]);
    frame.accel[1] = (int16_t)(raw[3]  << 8 | raw[2]);
    frame.accel[2] = (int16_t)(raw[5]  << 8 | raw[4]);
    frame.gyro[0]  = (int16_t)(raw[7]  << 8 | raw[6]);
    frame.gyro[1]  = (int16_t)(raw[9]  << 8 | raw[8]);
    frame.gyro[2]  = (int16_t)(raw[11] << 8 | raw[10]);

    /* Push to ring buffer */
    uint32_t next = (ring_head + 1) % IMU_RING_SIZE;
    if (next != ring_tail) {
        ring[ring_head] = frame;
        ring_head = next;
    }
}

bool imu_available(void)
{
    return ring_head != ring_tail;
}

bool imu_read(imu_frame_t *frame)
{
    if (ring_head == ring_tail)
        return false;
    *frame = ring[ring_tail];
    ring_tail = (ring_tail + 1) % IMU_RING_SIZE;
    return true;
}
