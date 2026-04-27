/**
 * system_clock.c — HSI16 → PLL → 170MHz, HSI48 + CRS for USB
 *
 * Crystal-less design: the CRS (Clock Recovery System) trims HSI48 using
 * USB SOF packets from the host, giving <±0.25% accuracy for USB.
 *
 * Init order matters: HSI48 → CRS → CLK48SEL = HSI48 → USB peripheral init.
 * If CRS is configured after USB init, HSI48 drift causes enumeration failures.
 */

#include "stm32g4xx_hal.h"
#include "config.h"

void system_clock_init(void)
{
    HAL_Init();

    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    /* ── Configure HSI16 → PLL → 170 MHz SYSCLK ────────────────────── */

    /* Enable power clock and set voltage scaling for 170 MHz */
    __HAL_RCC_PWR_CLK_ENABLE();
    HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST);

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI | RCC_OSCILLATORTYPE_HSI48;
    osc.HSIState = RCC_HSI_ON;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.HSI48State = RCC_HSI48_ON;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    /* HSI16 (16 MHz) / 4 = 4 MHz VCO input */
    osc.PLL.PLLM = RCC_PLLM_DIV4;
    /* 4 MHz × 85 = 340 MHz VCO */
    osc.PLL.PLLN = 85;
    /* VCO / 2 = 170 MHz PLLCLK */
    osc.PLL.PLLP = RCC_PLLP_DIV2;
    osc.PLL.PLLQ = RCC_PLLQ_DIV2;
    osc.PLL.PLLR = RCC_PLLR_DIV2;
    HAL_RCC_OscConfig(&osc);

    /* ── Configure system clocks ────────────────────────────────────── */

    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;   /* HCLK = 170 MHz */
    clk.APB1CLKDivider = RCC_HCLK_DIV1;    /* APB1 = 170 MHz */
    clk.APB2CLKDivider = RCC_HCLK_DIV1;    /* APB2 = 170 MHz */
    /* 4 wait states for 170 MHz at 1.28V (boost mode) */
    HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_4);

    /* ── Configure CRS to trim HSI48 from USB SOF ───────────────────── */

    __HAL_RCC_CRS_CLK_ENABLE();

    RCC_CRSInitTypeDef crs = {0};
    crs.Prescaler = RCC_CRS_SYNC_DIV1;
    crs.Source = RCC_CRS_SYNC_SOURCE_USB;
    crs.Polarity = RCC_CRS_SYNC_POLARITY_RISING;
    crs.ReloadValue = __HAL_RCC_CRS_RELOADVALUE_CALCULATE(HSI48_FREQ_HZ, 1000);
    crs.ErrorLimitValue = RCC_CRS_ERRORLIMIT_DEFAULT;
    crs.HSI48CalibrationValue = RCC_CRS_HSI48CALIBRATION_DEFAULT;
    HAL_RCCEx_CRSConfig(&crs);

    /* ── Select HSI48 as USB clock source ───────────────────────────── */

    RCC_PeriphCLKInitTypeDef pclk = {0};
    pclk.PeriphClockSelection = RCC_PERIPHCLK_USB;
    pclk.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;
    HAL_RCCEx_PeriphCLKConfig(&pclk);

    /* ── SysTick at 1ms for HAL_Delay and timestamping ──────────────── */
    HAL_SYSTICK_Config(SYSCLK_FREQ_HZ / 1000);
    HAL_SYSTICK_CLKSourceConfig(SYSTICK_CLKSOURCE_HCLK);
}

/**
 * Microsecond timestamp using DWT cycle counter with overflow extension.
 *
 * Raw DWT->CYCCNT wraps every 2^32 / 170e6 = 25.26 seconds. We detect
 * overflows in timestamp_update() (called from SysTick at 1kHz) and
 * maintain a 64-bit cycle count. The uint32_t microsecond result then
 * wraps at ~71 minutes (2^32 µs) — long enough for any recording session.
 */
static volatile uint32_t overflow_count = 0;
static volatile uint32_t last_cyccnt = 0;

void timestamp_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    overflow_count = 0;
    last_cyccnt = 0;
}

void timestamp_update(void)
{
    uint32_t now = DWT->CYCCNT;
    if (now < last_cyccnt)
        overflow_count++;
    last_cyccnt = now;
}

uint32_t timestamp_us(void)
{
    uint32_t ovf = overflow_count;
    uint32_t cyc = DWT->CYCCNT;
    /* Re-read overflow if CYCCNT wrapped between the two reads */
    if (cyc < last_cyccnt)
        ovf = overflow_count;
    uint64_t total = ((uint64_t)ovf << 32) | cyc;
    return (uint32_t)(total / (SYSCLK_FREQ_HZ / 1000000U));
}

/* ── SysTick handler: HAL tick + DWT overflow tracking ───────────────── */

void SysTick_Handler(void)
{
    HAL_IncTick();
    timestamp_update();
}
