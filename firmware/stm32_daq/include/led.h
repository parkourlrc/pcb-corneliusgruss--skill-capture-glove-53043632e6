/**
 * led.h — LED_REC (PB4) and LED_SYS (PB5) control
 */

#ifndef LED_H
#define LED_H

#include <stdbool.h>

void led_init(void);

void led_rec_set(bool on);
void led_rec_toggle(void);

void led_sys_set(bool on);
void led_sys_toggle(void);

#endif /* LED_H */
