/**
 * stm32g4xx_hal_conf.h — HAL module selection for Glove DAQ
 *
 * Enable only the HAL modules actually used to minimize code size.
 */

#ifndef STM32G4XX_HAL_CONF_H
#define STM32G4XX_HAL_CONF_H

/* ── Module selection ─────────────────────────────────────────────────── */
#define HAL_MODULE_ENABLED
#define HAL_ADC_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_EXTI_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_I2C_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_SPI_MODULE_ENABLED
#define HAL_TIM_MODULE_ENABLED

/* ── Oscillator values ────────────────────────────────────────────────── */
#if !defined(HSI_VALUE)
  #define HSI_VALUE    16000000U  /* HSI = 16 MHz */
#endif
#if !defined(HSI48_VALUE)
  #define HSI48_VALUE  48000000U  /* HSI48 = 48 MHz */
#endif
#if !defined(LSI_VALUE)
  #define LSI_VALUE    32000U     /* LSI = 32 kHz */
#endif
#if !defined(HSE_VALUE)
  #define HSE_VALUE    8000000U   /* Not used (crystal-less design) */
#endif
#if !defined(LSE_VALUE)
  #define LSE_VALUE    32768U     /* Not used */
#endif
#if !defined(EXTERNAL_CLOCK_VALUE)
  #define EXTERNAL_CLOCK_VALUE 12288000U
#endif

/* ── HSE/LSE startup timeouts ─────────────────────────────────────────── */
#if !defined(HSE_STARTUP_TIMEOUT)
  #define HSE_STARTUP_TIMEOUT  100U
#endif
#if !defined(LSE_STARTUP_TIMEOUT)
  #define LSE_STARTUP_TIMEOUT  5000U
#endif

/* ── SysTick ──────────────────────────────────────────────────────────── */
#define TICK_INT_PRIORITY      15U  /* Lowest priority for HAL tick */
#define USE_RTOS               0U
#define PREFETCH_ENABLE        1U
#define INSTRUCTION_CACHE_ENABLE  1U
#define DATA_CACHE_ENABLE      1U

/* ── HAL module includes ──────────────────────────────────────────────── */
#ifdef HAL_RCC_MODULE_ENABLED
  #include "stm32g4xx_hal_rcc.h"
  #include "stm32g4xx_hal_rcc_ex.h"
#endif
#ifdef HAL_GPIO_MODULE_ENABLED
  #include "stm32g4xx_hal_gpio.h"
  #include "stm32g4xx_hal_gpio_ex.h"
#endif
#ifdef HAL_DMA_MODULE_ENABLED
  #include "stm32g4xx_hal_dma.h"
  #include "stm32g4xx_hal_dma_ex.h"
#endif
#ifdef HAL_CORTEX_MODULE_ENABLED
  #include "stm32g4xx_hal_cortex.h"
#endif
#ifdef HAL_ADC_MODULE_ENABLED
  #include "stm32g4xx_hal_adc.h"
  #include "stm32g4xx_hal_adc_ex.h"
#endif
#ifdef HAL_EXTI_MODULE_ENABLED
  #include "stm32g4xx_hal_exti.h"
#endif
#ifdef HAL_FLASH_MODULE_ENABLED
  #include "stm32g4xx_hal_flash.h"
  #include "stm32g4xx_hal_flash_ex.h"
  #include "stm32g4xx_hal_flash_ramfunc.h"
#endif
#ifdef HAL_I2C_MODULE_ENABLED
  #include "stm32g4xx_hal_i2c.h"
  #include "stm32g4xx_hal_i2c_ex.h"
#endif
#ifdef HAL_PWR_MODULE_ENABLED
  #include "stm32g4xx_hal_pwr.h"
  #include "stm32g4xx_hal_pwr_ex.h"
#endif
#ifdef HAL_SPI_MODULE_ENABLED
  #include "stm32g4xx_hal_spi.h"
  #include "stm32g4xx_hal_spi_ex.h"
#endif
#ifdef HAL_TIM_MODULE_ENABLED
  #include "stm32g4xx_hal_tim.h"
  #include "stm32g4xx_hal_tim_ex.h"
#endif

/* ── Assert macro ─────────────────────────────────────────────────────── */
/* #define USE_FULL_ASSERT 1 */
#ifdef USE_FULL_ASSERT
  #define assert_param(expr) ((expr) ? (void)0U : assert_failed((uint8_t *)__FILE__, __LINE__))
  void assert_failed(uint8_t *file, uint32_t line);
#else
  #define assert_param(expr) ((void)0U)
#endif

#endif /* STM32G4XX_HAL_CONF_H */
