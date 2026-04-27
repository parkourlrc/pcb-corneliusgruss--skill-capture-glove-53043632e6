/**
 * sync.c — 30Hz camera sync pulse via TIM1_CH1 (PA8)
 *
 * TIM1 generates a 30Hz PWM output that triggers both stereo cameras.
 * The capture/compare interrupt records a precise timestamp at each pulse,
 * which gets embedded in the proprioceptive packet for camera-sensor
 * time correlation.
 */

#include "sync.h"
#include "config.h"
#include "stm32g4xx_hal.h"

static TIM_HandleTypeDef htim;
static volatile uint32_t last_sync_us = 0;
static volatile uint32_t sync_count = 0;
static uint32_t last_read_count = 0;

extern uint32_t timestamp_us(void);

void sync_init(void)
{
    SYNC_TIM_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* Configure PA8 as TIM1_CH1 alternate function */
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = SYNC_GPIO_PIN;
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    gpio.Alternate = SYNC_GPIO_AF;
    HAL_GPIO_Init(SYNC_GPIO_PORT, &gpio);

    /*
     * TIM1 clock = 170 MHz (APB2)
     * Target: 30 Hz PWM with 50% duty cycle
     *
     * Prescaler: 170 → timer clock = 1 MHz
     * Period: 1000000 / 30 = 33333
     * Pulse: 33333 / 2 = 16666 (50% duty)
     */
    htim.Instance = SYNC_TIM;
    htim.Init.Prescaler = (SYSCLK_FREQ_HZ / 1000000U) - 1;  /* → 1 MHz */
    htim.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim.Init.Period = (1000000U / SYNC_FREQ_HZ) - 1;        /* → 33333 */
    htim.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim.Init.RepetitionCounter = 0;
    htim.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
    HAL_TIM_PWM_Init(&htim);

    /* PWM channel configuration */
    TIM_OC_InitTypeDef oc = {0};
    oc.OCMode = TIM_OCMODE_PWM1;
    oc.Pulse = ((1000000U / SYNC_FREQ_HZ) - 1) / 2;  /* 50% duty */
    oc.OCPolarity = TIM_OCPOLARITY_HIGH;
    oc.OCNPolarity = TIM_OCNPOLARITY_HIGH;
    oc.OCFastMode = TIM_OCFAST_DISABLE;
    oc.OCIdleState = TIM_OCIDLESTATE_RESET;
    HAL_TIM_PWM_ConfigChannel(&htim, &oc, TIM_CHANNEL_1);

    /* Enable capture/compare interrupt for timestamping */
    HAL_NVIC_SetPriority(SYNC_TIM_IRQn, 2, 0);
    HAL_NVIC_EnableIRQ(SYNC_TIM_IRQn);
}

void sync_start(void)
{
    HAL_TIM_PWM_Start_IT(&htim, TIM_CHANNEL_1);

    /* TIM1 requires MOE (Main Output Enable) for advanced timer */
    __HAL_TIM_MOE_ENABLE(&htim);
}

void sync_stop(void)
{
    HAL_TIM_PWM_Stop_IT(&htim, TIM_CHANNEL_1);
}

uint32_t sync_get_timestamp(void)
{
    if (sync_count == last_read_count)
        return 0;

    last_read_count = sync_count;
    return last_sync_us;
}

uint16_t sync_get_count(void)
{
    return (uint16_t)sync_count;
}

/* ── ISR ──────────────────────────────────────────────────────────────── */

void sync_irq_handler(void)
{
    last_sync_us = timestamp_us();
    sync_count++;
}

void TIM1_CC_IRQHandler(void)
{
    if (__HAL_TIM_GET_FLAG(&htim, TIM_FLAG_CC1) &&
        __HAL_TIM_GET_IT_SOURCE(&htim, TIM_IT_CC1)) {
        __HAL_TIM_CLEAR_IT(&htim, TIM_IT_CC1);
        sync_irq_handler();
    }
}

/* TIM1 is an advanced timer — HAL may enable update/break/trigger interrupts.
 * All must have handlers or they hit Default_Handler → infinite loop. */
void TIM1_UP_TIM16_IRQHandler(void)
{
    __HAL_TIM_CLEAR_IT(&htim, TIM_IT_UPDATE);
}

void TIM1_BRK_TIM15_IRQHandler(void)
{
    __HAL_TIM_CLEAR_IT(&htim, TIM_IT_BREAK);
}

void TIM1_TRG_COM_TIM17_IRQHandler(void)
{
    __HAL_TIM_CLEAR_IT(&htim, TIM_IT_TRIGGER | TIM_IT_COM);
}
