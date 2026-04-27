/**
 * tof.h — VL53L1X Time-of-Flight distance sensor driver
 */

#ifndef TOF_H
#define TOF_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Initialize a VL53L1X on the given mux channel.
 * Sets timing budget to 20ms for 30Hz operation.
 */
bool tof_init(uint8_t mux_channel);

/**
 * Start a ranging measurement on the given mux channel.
 */
bool tof_start_ranging(uint8_t mux_channel);

/**
 * Read the latest distance in mm from the given mux channel.
 * Returns 0xFFFF if no new data is available.
 */
uint16_t tof_read_distance(uint8_t mux_channel);

/**
 * Read both ToF sensors. distances[0] = top, distances[1] = bottom.
 * Returns 0xFFFF for sensors with no new reading.
 */
void tof_read_all(uint16_t distances[2]);

/**
 * Get consecutive failure count for a sensor (0 or 1).
 */
uint8_t tof_get_fail_count(uint8_t sensor_idx);

/**
 * Get last successfully read distance for a sensor (0 or 1).
 */
uint16_t tof_get_last_good(uint8_t sensor_idx);

#endif /* TOF_H */
