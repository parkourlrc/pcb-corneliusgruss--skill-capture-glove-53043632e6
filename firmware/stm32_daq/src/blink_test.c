/**
 * blink_test.c — IMU diagnostic: tries SPI then I2C on same pins
 */

#include "stm32g4xx_hal.h"
#include "config.h"
#include "tusb.h"
#include <stdio.h>
#include <string.h>

extern void system_clock_init(void);

static void led_gpio_init(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = LED_SYS_PIN | LED_REC_PIN;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED_SYS_PORT, &gpio);
}

/* USB */
void USB_HP_IRQHandler(void)    { tud_int_handler(0); }
void USB_LP_IRQHandler(void)    { tud_int_handler(0); }
void USBWakeUp_IRQHandler(void) { tud_int_handler(0); }
void tud_mount_cb(void) {}
void tud_umount_cb(void) {}
void tud_suspend_cb(bool rw) { (void)rw; }
void tud_resume_cb(void) {}
void tud_cdc_line_state_cb(uint8_t i, bool d, bool r) { (void)i;(void)d;(void)r; }
void tud_cdc_rx_cb(uint8_t i) { (void)i; }

static void usb_init(void)
{
    __HAL_RCC_PWR_CLK_ENABLE();
    PWR->CR3 |= PWR_CR3_UCPD_DBDIS;
    __HAL_RCC_USB_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    HAL_NVIC_SetPriority(USB_HP_IRQn, 3, 0);
    HAL_NVIC_SetPriority(USB_LP_IRQn, 3, 0);
    HAL_NVIC_SetPriority(USBWakeUp_IRQn, 3, 0);
    tusb_init();
}

static char pbuf[128];
static void cdc_print(const char *s)
{
    if (!tud_cdc_connected()) return;
    tud_cdc_write(s, strlen(s));
    tud_cdc_write_flush();
    /* Give USB time to drain */
    for (int i = 0; i < 10; i++) { tud_task(); HAL_Delay(1); }
}

/* ── Bit-bang I2C on PA5(SCL) + PA7(SDA), PA4(CS) held high ──────── */

#define PIN_CS   GPIO_PIN_4
#define PIN_SCL  GPIO_PIN_5
#define PIN_SDA  GPIO_PIN_7

static void i2c_delay(void) { for (volatile int i = 0; i < 40; i++) {} }

static void i2c_gpio_init(void)
{
    __HAL_RCC_GPIOA_CLK_ENABLE();
    GPIO_InitTypeDef gpio = {0};

    /* CS high = I2C mode on ICM-42688 */
    gpio.Pin = PIN_CS;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &gpio);
    GPIOA->BSRR = PIN_CS;  /* CS HIGH */

    /* SCL + SDA as open-drain with pull-ups */
    gpio.Pin = PIN_SCL | PIN_SDA;
    gpio.Mode = GPIO_MODE_OUTPUT_OD;
    gpio.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOA, &gpio);

    GPIOA->BSRR = PIN_SCL;
    GPIOA->BSRR = PIN_SDA;
    HAL_Delay(1);
}

static void i2c_start(void)
{
    GPIOA->BSRR = PIN_SDA; i2c_delay();
    GPIOA->BSRR = PIN_SCL; i2c_delay();
    GPIOA->BRR  = PIN_SDA; i2c_delay();
    GPIOA->BRR  = PIN_SCL; i2c_delay();
}

static void i2c_stop(void)
{
    GPIOA->BRR  = PIN_SDA; i2c_delay();
    GPIOA->BSRR = PIN_SCL; i2c_delay();
    GPIOA->BSRR = PIN_SDA; i2c_delay();
}

static int i2c_write_byte(uint8_t b)
{
    for (int i = 7; i >= 0; i--) {
        if (b & (1 << i)) GPIOA->BSRR = PIN_SDA; else GPIOA->BRR = PIN_SDA;
        i2c_delay();
        GPIOA->BSRR = PIN_SCL; i2c_delay();
        GPIOA->BRR  = PIN_SCL; i2c_delay();
    }
    /* Read ACK */
    GPIOA->BSRR = PIN_SDA; /* release */
    i2c_delay();
    GPIOA->BSRR = PIN_SCL; i2c_delay();
    int ack = (GPIOA->IDR & PIN_SDA) ? 1 : 0;
    GPIOA->BRR = PIN_SCL; i2c_delay();
    return ack; /* 0=ACK, 1=NACK */
}

static uint8_t i2c_read_byte(int send_ack)
{
    uint8_t val = 0;
    GPIOA->BSRR = PIN_SDA; /* release SDA for input */
    for (int i = 7; i >= 0; i--) {
        i2c_delay();
        GPIOA->BSRR = PIN_SCL; i2c_delay();
        if (GPIOA->IDR & PIN_SDA) val |= (1 << i);
        GPIOA->BRR = PIN_SCL; i2c_delay();
    }
    /* Send ACK/NACK */
    if (send_ack) GPIOA->BRR = PIN_SDA; else GPIOA->BSRR = PIN_SDA;
    i2c_delay();
    GPIOA->BSRR = PIN_SCL; i2c_delay();
    GPIOA->BRR  = PIN_SCL; i2c_delay();
    GPIOA->BSRR = PIN_SDA;
    return val;
}

static int i2c_read_reg(uint8_t dev_addr, uint8_t reg, uint8_t *val)
{
    i2c_start();
    if (i2c_write_byte(dev_addr << 1)) { i2c_stop(); return -1; } /* NACK */
    if (i2c_write_byte(reg))           { i2c_stop(); return -2; }
    i2c_start(); /* repeated start */
    if (i2c_write_byte((dev_addr << 1) | 1)) { i2c_stop(); return -3; }
    *val = i2c_read_byte(0); /* NACK after last byte */
    i2c_stop();
    return 0;
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void)
{
    system_clock_init();
    led_gpio_init();
    usb_init();

    HAL_GPIO_WritePin(LED_SYS_PORT, LED_SYS_PIN, GPIO_PIN_SET);
    while (!tud_cdc_connected()) { tud_task(); }
    HAL_Delay(200);

    cdc_print("=== IMU I2C Probe (CS=HIGH) ===\r\n");

    i2c_gpio_init();
    HAL_Delay(100);

    /* Scan I2C addresses 0x00-0x7F */
    cdc_print("I2C bus scan:\r\n");
    int found = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        i2c_start();
        int ack = i2c_write_byte(addr << 1);
        i2c_stop();
        HAL_Delay(1);
        if (ack == 0) {
            snprintf(pbuf, sizeof(pbuf), "  ACK at 0x%02X\r\n", addr);
            cdc_print(pbuf);
            found++;
        }
    }
    if (!found) cdc_print("  No devices found\r\n");

    /* Read WHO_AM_I from 0x6A at multiple possible register addresses */
    uint8_t val = 0;
    uint8_t who_regs[] = {0x0F, 0x00, 0x01, 0x75, 0x76};
    for (int i = 0; i < 5; i++) {
        int rc = i2c_read_reg(0x6A, who_regs[i], &val);
        snprintf(pbuf, sizeof(pbuf), "I2C 0x6A REG[0x%02X] = 0x%02X (rc=%d)\r\n", who_regs[i], val, rc);
        cdc_print(pbuf);
    }
    /* Dump first 32 registers */
    cdc_print("Reg dump 0x6A:\r\n");
    for (int r = 0; r < 0x80; r += 8) {
        snprintf(pbuf, sizeof(pbuf), "  0x%02X:", r);
        cdc_print(pbuf);
        for (int j = 0; j < 8; j++) {
            i2c_read_reg(0x6A, r + j, &val);
            snprintf(pbuf, sizeof(pbuf), " %02X", val);
            cdc_print(pbuf);
        }
        cdc_print("\r\n");
    }

    /* ── Configure QMI8658C at I2C 0x6A ────────────────────────────── */
    cdc_print("\r\nChip is QMI8658C (WHO_AM_I=0x05 at reg 0x00)\r\n");
    cdc_print("Configuring QMI8658C...\r\n");

    #define i2c_write_reg(addr, reg, v) do { \
        i2c_start(); \
        i2c_write_byte((addr) << 1); \
        i2c_write_byte(reg); \
        i2c_write_byte(v); \
        i2c_stop(); \
        HAL_Delay(1); \
    } while(0)

    #define QMI_ADDR  0x6A
    #define QMI_CTRL1 0x02
    #define QMI_CTRL2 0x03  /* Accel config: [7:4]=self-test, [3:0]= ODR+range */
    #define QMI_CTRL3 0x04  /* Gyro config */
    #define QMI_CTRL5 0x06  /* Low-pass filter */
    #define QMI_CTRL7 0x08  /* Enable: bit0=accel, bit1=gyro */

    /* CTRL2: Accel ±2g (bits[6:4]=000), ODR 1kHz (bits[3:0]=0x07) */
    i2c_write_reg(QMI_ADDR, QMI_CTRL2, 0x07);
    /* CTRL3: Gyro ±256dps (bits[6:4]=010), ODR 1kHz (bits[3:0]=0x07) */
    i2c_write_reg(QMI_ADDR, QMI_CTRL3, 0x27);
    /* CTRL7: Enable accel (bit0) + gyro (bit1) */
    i2c_write_reg(QMI_ADDR, QMI_CTRL7, 0x03);
    HAL_Delay(50);

    /* Verify */
    i2c_read_reg(QMI_ADDR, QMI_CTRL7, &val);
    snprintf(pbuf, sizeof(pbuf), "CTRL7 readback: 0x%02X (expect 0x03)\r\n", val);
    cdc_print(pbuf);

    cdc_print("=== Streaming (QMI8658 regs 0x35-0x40) ===\r\n");

    uint32_t last = 0;
    bool state = false;
    while (1) {
        tud_task();
        uint32_t now = HAL_GetTick();
        if (now - last >= 200) {
            last = now;
            state = !state;
            HAL_GPIO_WritePin(LED_SYS_PORT, LED_SYS_PIN, state ? GPIO_PIN_SET : GPIO_PIN_RESET);

            /* QMI8658 data: accel 0x35-0x3A, gyro 0x3B-0x40 (little-endian!) */
            uint8_t raw[12];
            for (int i = 0; i < 12; i++)
                i2c_read_reg(QMI_ADDR, 0x35 + i, &raw[i]);

            /* QMI8658 is LITTLE-endian (low byte first) */
            int16_t ax = (int16_t)(raw[1]<<8 | raw[0]);
            int16_t ay = (int16_t)(raw[3]<<8 | raw[2]);
            int16_t az = (int16_t)(raw[5]<<8 | raw[4]);
            int16_t gx = (int16_t)(raw[7]<<8 | raw[6]);
            int16_t gy = (int16_t)(raw[9]<<8 | raw[8]);
            int16_t gz = (int16_t)(raw[11]<<8 | raw[10]);

            snprintf(pbuf, sizeof(pbuf),
                     "ax=%6d ay=%6d az=%6d gx=%6d gy=%6d gz=%6d\r\n",
                     ax, ay, az, gx, gy, gz);
            cdc_print(pbuf);
        }
    }
}
