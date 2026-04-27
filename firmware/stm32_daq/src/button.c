/**
 * button.c — SW_RECORD (PB3) with software debounce
 *
 * Active low (internal pull-up). Debounce window: 20ms.
 */

#include "button.h"
#include "config.h"
#include "stm32g4xx_hal.h"

#define DEBOUNCE_MS 20

static bool debounced_state = false;
static bool prev_debounced = false;
static bool edge_flag = false;
static uint32_t last_change_tick = 0;

void button_init(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();

    GPIO_InitTypeDef gpio = {0};
    gpio.Pin = BTN_REC_PIN;
    gpio.Mode = GPIO_MODE_INPUT;
    gpio.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(BTN_REC_PORT, &gpio);
}

void button_update(void)
{
    bool raw = (HAL_GPIO_ReadPin(BTN_REC_PORT, BTN_REC_PIN) == GPIO_PIN_RESET);
    uint32_t now = HAL_GetTick();

    if (raw != debounced_state) {
        if ((now - last_change_tick) >= DEBOUNCE_MS) {
            prev_debounced = debounced_state;
            debounced_state = raw;
            last_change_tick = now;

            /* Detect rising edge (button press) */
            if (debounced_state && !prev_debounced)
                edge_flag = true;
        }
    } else {
        last_change_tick = now;
    }
}

bool button_is_pressed(void)
{
    return debounced_state;
}

bool button_was_pressed(void)
{
    if (edge_flag) {
        edge_flag = false;
        return true;
    }
    return false;
}
