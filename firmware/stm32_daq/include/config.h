/**
 * config.h — Pin definitions, I2C addresses, and timing constants
 *
 * All hardware-specific constants for the Glove DAQ board (v2.0 Rev A).
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "stm32g4xx_hal.h"

/* ── Clock ─────────────────────────────────────────────────────────────── */
#define SYSCLK_FREQ_HZ         170000000U
#define HSI48_FREQ_HZ          48000000U

/* ── SPI1 — QMI8658C IMU (JLCPCB substitution for ICM-42688-P) ────────── */
#define IMU_SPI                 SPI1
#define IMU_SPI_CLK_ENABLE()    __HAL_RCC_SPI1_CLK_ENABLE()
#define IMU_SPI_GPIO_PORT       GPIOA
#define IMU_SPI_GPIO_CLK_ENABLE() __HAL_RCC_GPIOA_CLK_ENABLE()
#define IMU_SCK_PIN             GPIO_PIN_5
#define IMU_MISO_PIN            GPIO_PIN_6
#define IMU_MOSI_PIN            GPIO_PIN_7
#define IMU_SPI_AF              GPIO_AF5_SPI1
#define IMU_SPI_PRESCALER       SPI_BAUDRATEPRESCALER_16  /* 170/16 ≈ 10.6 MHz */

/* IMU chip select — manual GPIO, active low */
#define IMU_CS_PORT             GPIOA
#define IMU_CS_PIN              GPIO_PIN_4

/* IMU data-ready interrupt — PB0, EXTI0 */
#define IMU_INT_PORT            GPIOB
#define IMU_INT_PIN             GPIO_PIN_0
#define IMU_INT_EXTI_IRQn       EXTI0_IRQn

/* ── I2C1 — TCA9548A mux + all I2C sensors ───────────────────────────── */
#define SENSOR_I2C              I2C1
#define SENSOR_I2C_CLK_ENABLE() __HAL_RCC_I2C1_CLK_ENABLE()
#define SENSOR_I2C_GPIO_PORT    GPIOB
#define SENSOR_I2C_GPIO_CLK_ENABLE() __HAL_RCC_GPIOB_CLK_ENABLE()
#define SENSOR_I2C_SCL_PIN      GPIO_PIN_6
#define SENSOR_I2C_SDA_PIN      GPIO_PIN_7
#define SENSOR_I2C_AF           GPIO_AF4_I2C1

/* I2C speed: 400 kHz (Fast Mode) */
#define SENSOR_I2C_TIMING       0x10802D9BU  /* 170 MHz, Fast Mode 400 kHz */

/* ── I2C Mux — TCA9548A ──────────────────────────────────────────────── */
#define MUX_I2C_ADDR            (0x70 << 1)  /* 7-bit 0x70, shifted for HAL */

/* Channel assignment — verified against physical wiring. */
#define MUX_CH_TOF_BOT          0   /* CH0 */
#define MUX_CH_HALL_3F_MCP      1   /* CH1 — MLX90393 joint sensor */
#define MUX_CH_HALL_INDEX_PIP   2   /* CH2 — MLX90393 joint sensor */
#define MUX_CH_HALL_3F_PIP      3   /* CH3 — MLX90393 joint sensor */
#define MUX_CH_HALL_INDEX_MCP   4   /* CH4 — MLX90393 joint sensor */
#define MUX_CH_TOF_TOP          5   /* CH5 */

/* ── Sensor I2C Addresses (7-bit, shifted for HAL) ────────────────────── */
#define MLX90393_I2C_ADDR       (0x0C << 1)
#define VL53L1X_I2C_ADDR        (0x29 << 1)

/* ── MLX90393 Commands ────────────────────────────────────────────────── */
#define MLX90393_CMD_SM_XYZ     0x3E  /* Start Measurement: X+Y+Z+T */
#define MLX90393_CMD_RM_XYZ     0x4E  /* Read Measurement: X+Y+Z+T */
#define MLX90393_CMD_WR         0x60  /* Write Register */
#define MLX90393_CMD_RR         0x50  /* Read Register */
#define MLX90393_CMD_EX         0x80  /* Exit mode */
#define MLX90393_CMD_RT         0xF0  /* Reset */

/* MLX90393 register addresses */
#define MLX90393_REG_GAIN_SEL   0x00  /* GAIN_SEL in register 0, bits [7:4] */
#define MLX90393_REG_RES_XYZ    0x02  /* Resolution bits */

/* GAIN_SEL applied to every Hall sensor at boot. 0 = max gain / narrow range,
 * 7 = min gain / widest range. Our finger magnets saturated the chip at NVRAM
 * defaults, so we start at 7 and step up if the signal swing is too small. */
#define MLX90393_GAIN_SEL       7

/* ── VL53L1X ──────────────────────────────────────────────────────────── */
#define VL53L1X_TIMING_BUDGET_MS 20   /* 20ms for 30Hz operation */

/* ── TIM1 — Camera sync pulse ────────────────────────────────────────── */
#define SYNC_TIM                TIM1
#define SYNC_TIM_CLK_ENABLE()   __HAL_RCC_TIM1_CLK_ENABLE()
#define SYNC_GPIO_PORT          GPIOA
#define SYNC_GPIO_PIN           GPIO_PIN_8
#define SYNC_GPIO_AF            GPIO_AF6_TIM1
#define SYNC_FREQ_HZ            30U
#define SYNC_TIM_IRQn           TIM1_CC_IRQn

/* ── UI — LEDs and button ─────────────────────────────────────────────── */
#define LED_REC_PORT            GPIOB
#define LED_REC_PIN             GPIO_PIN_4
#define LED_SYS_PORT            GPIOB
#define LED_SYS_PIN             GPIO_PIN_5

#define BTN_REC_PORT            GPIOB
#define BTN_REC_PIN             GPIO_PIN_3

/* ── Timing ───────────────────────────────────────────────────────────── */
#define IMU_RATE_HZ             1000U
#define PROPRIO_RATE_HZ         30U   /* Match camera frame rate — policy only sees 30Hz */
#define PROPRIO_PERIOD_US       (1000000U / PROPRIO_RATE_HZ)

/* ── USB Packet Types ─────────────────────────────────────────────────── */
#define PKT_TYPE_IMU            0x01
#define PKT_TYPE_PROPRIO        0x02

#define PKT_SIZE_IMU            17  /* 1 type + 4 ts + 6 accel + 6 gyro */
#define PKT_SIZE_PROPRIO        40  /* 1 + 4 + 24 (4× Hall xyz) + 4 (2× ToF) + 4 sync_ts + 1 status + 2 sync_seq */

/* ── Ring Buffer Sizes ────────────────────────────────────────────────── */
#define IMU_RING_SIZE           64    /* IMU frames buffered before USB drain */
#define USB_TX_BUF_SIZE         8192  /* USB CDC transmit buffer (200Hz IMU + 100Hz proprio) */

/* ── Sensor Counts ────────────────────────────────────────────────────── */
#define NUM_HALL_SENSORS        4
#define NUM_TOF_SENSORS         2
#define HALL_DATA_SLOTS         4

/* ── DMA ──────────────────────────────────────────────────────────────── */
#define IMU_SPI_TX_DMA          DMA1_Channel1
#define IMU_SPI_RX_DMA          DMA1_Channel2
#define IMU_DMA_CLK_ENABLE()    __HAL_RCC_DMA1_CLK_ENABLE(); __HAL_RCC_DMAMUX1_CLK_ENABLE()

/* ── Status Flag Bits (packet byte 45) ────────────────────────────────── */
#define STATUS_RECORDING        (1U << 0)
#define STATUS_SYNC_FIRED       (1U << 1)
#define STATUS_BTN_PRESSED      (1U << 2)

#endif /* CONFIG_H */
