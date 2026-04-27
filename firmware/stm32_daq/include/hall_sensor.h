/**
 * hall_sensor.h — MLX90393 3D Hall effect sensor driver
 *
 * Each sensor outputs raw Bx/By/Bz as int16. Angle and force decomposition
 * happens on the host, not on the MCU.
 *
 * I2C protocol: SM (Start Measurement) → wait ~1ms → RM (Read Measurement).
 */

#ifndef HALL_SENSOR_H
#define HALL_SENSOR_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int16_t bx;
    int16_t by;
    int16_t bz;
} hall_data_t;

/**
 * Reset and configure a Hall sensor on the given mux channel.
 * Verifies communication by reading back the gain register.
 */
bool hall_sensor_init(uint8_t mux_channel);

/**
 * Configure the GAIN_SEL field (0-7) for a sensor on the given mux channel.
 * Gain must be tuned per-sensor based on magnet distance.
 *
 * GAIN_SEL values (MLX90393 datasheet Table 18):
 *   0 = 5x (highest sensitivity), 7 = 1x (lowest sensitivity)
 */
bool hall_sensor_configure_gain(uint8_t mux_channel, uint8_t gain_sel);

/**
 * Read the current GAIN_SEL field (0-7) from the sensor on mux_channel.
 * Returns false on I2C error; *gain_out is unmodified in that case.
 */
bool hall_sensor_get_gain(uint8_t mux_channel, uint8_t *gain_out);

/**
 * Start a measurement on the given mux channel (sends SM command).
 * Call hall_sensor_read() after ~1ms to retrieve results.
 */
bool hall_sensor_start_measurement(uint8_t mux_channel);

/**
 * Read measurement results from the given mux channel (sends RM command).
 * Returns false if the sensor reports an error or data isn't ready.
 */
bool hall_sensor_read(uint8_t mux_channel, hall_data_t *data);

/**
 * Return the mux channel for Hall sensor slot i (0-3).
 */
uint8_t hall_sensor_channel(uint8_t slot);

/**
 * Convenience: start all Hall sensors, wait, read all.
 * data array must have HALL_DATA_SLOTS entries (unused slots zeroed).
 */
void hall_sensor_read_all(hall_data_t data[4]);

#endif /* HALL_SENSOR_H */
