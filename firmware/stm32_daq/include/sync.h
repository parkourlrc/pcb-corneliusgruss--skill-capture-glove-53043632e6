/**
 * sync.h — 30Hz camera sync pulse via TIM1_CH1 (PA8)
 *
 * Generates a PWM pulse that triggers both stereo cameras and records
 * a precise timestamp for correlating camera frames with sensor data.
 */

#ifndef SYNC_H
#define SYNC_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Initialize TIM1 CH1 as 30Hz PWM output on PA8 with capture interrupt.
 */
void sync_init(void);

/**
 * Start generating sync pulses.
 */
void sync_start(void);

/**
 * Stop generating sync pulses.
 */
void sync_stop(void);

/**
 * Get the timestamp (µs) of the last sync pulse. Returns 0 if no pulse
 * has fired since the last call (automatically cleared after read).
 */
uint32_t sync_get_timestamp(void);

/**
 * Get the current sync pulse count (monotonically increasing).
 * Unlike sync_get_timestamp(), this does not clear on read.
 */
uint16_t sync_get_count(void);

/**
 * Called from TIM1 capture/compare ISR.
 */
void sync_irq_handler(void);

#endif /* SYNC_H */
