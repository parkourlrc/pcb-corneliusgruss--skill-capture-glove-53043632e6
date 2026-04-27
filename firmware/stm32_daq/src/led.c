/**
 * led.c — LED_REC (PB4) and LED_SYS (PB5) control
 */

#include "led.h"
#include "config.h"
#include "stm32g4xx_hal.h"

void led_init(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();

    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = LED_REC_PIN | LED_SYS_PIN;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &gpio);

    /* Start with both LEDs off */
    HAL_GPIO_WritePin(LED_REC_PORT, LED_REC_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_SYS_PORT, LED_SYS_PIN, GPIO_PIN_RESET);
}

void led_rec_set(bool on)
{
    HAL_GPIO_WritePin(LED_REC_PORT, LED_REC_PIN,
                      on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void led_rec_toggle(void)
{
    HAL_GPIO_TogglePin(LED_REC_PORT, LED_REC_PIN);
}

void led_sys_set(bool on)
{
    HAL_GPIO_WritePin(LED_SYS_PORT, LED_SYS_PIN,
                      on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void led_sys_toggle(void)
{
    HAL_GPIO_TogglePin(LED_SYS_PORT, LED_SYS_PIN);
}
