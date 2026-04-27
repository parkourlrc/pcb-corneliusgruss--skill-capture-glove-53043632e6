/**
 * i2c_mux.h — TCA9548A I2C multiplexer driver
 */

#ifndef I2C_MUX_H
#define I2C_MUX_H

#include <stdint.h>
#include <stdbool.h>
#include "stm32g4xx_hal.h"

/**
 * Initialize I2C1 peripheral only (GPIO + 400 kHz Fast Mode).
 * Must be called before any I2C operations.
 * Returns true if the peripheral initialized successfully.
 */
bool i2c_bus_init(void);

/**
 * Probe the TCA9548A mux at address 0x70.
 * Requires i2c_bus_init() to have been called first.
 * Returns true if the mux is present and responding.
 */
bool i2c_mux_probe(void);

/**
 * Legacy init: calls i2c_bus_init() + i2c_mux_probe().
 */
bool i2c_mux_init(void);

/**
 * Select a mux channel (0-7). Only one channel active at a time.
 * Returns true on success.
 */
bool i2c_mux_select(uint8_t channel);

/**
 * Disable all mux channels.
 */
bool i2c_mux_disable(void);

/**
 * Scan I2C bus for all responding addresses (0x08-0x77).
 * Writes found addresses into `addrs` array, returns count.
 * Requires i2c_bus_init() to have been called first.
 */
uint8_t i2c_scan(uint8_t *addrs, uint8_t max_addrs);

/**
 * Bit-bang I2C register write. Caller must select mux channel first.
 * addr7 is the 7-bit device address (not shifted).
 */
bool i2c_bb_write_reg(uint8_t addr7, uint8_t reg, const uint8_t *data, uint8_t len);

/**
 * Bit-bang I2C register read. Caller must select mux channel first.
 * addr7 is the 7-bit device address (not shifted).
 */
bool i2c_bb_read_reg(uint8_t addr7, uint8_t reg, uint8_t *data, uint8_t len);

/**
 * Bit-bang I2C write with 16-bit register address (for VL53L1X etc).
 */
bool i2c_bb_write_reg16(uint8_t addr7, uint16_t reg, const uint8_t *data, uint8_t len);

/**
 * Bit-bang I2C read with 16-bit register address (for VL53L1X etc).
 */
bool i2c_bb_read_reg16(uint8_t addr7, uint16_t reg, uint8_t *data, uint8_t len);

/**
 * Raw I2C write (no register address). For command-based devices like MLX90393.
 * Sends: START → addr+W → data[0..len-1] → STOP
 */
bool i2c_bb_write_raw(uint8_t addr7, const uint8_t *data, uint8_t len);

/**
 * Raw I2C read (no register address). For command-based devices like MLX90393.
 * Sends: START → addr+R → reads len bytes → STOP
 */
bool i2c_bb_read_raw(uint8_t addr7, uint8_t *data, uint8_t len);

/**
 * Write then read in a single I2C transaction using repeated start.
 * Sends: START → addr+W → tx[0..tx_len-1] → REPEATED START → addr+R → reads rx_len bytes → STOP
 * Required by devices like MLX90393 that expect repeated start between command and response.
 */
bool i2c_bb_write_read(uint8_t addr7, const uint8_t *tx, uint8_t tx_len, uint8_t *rx, uint8_t rx_len);

#endif /* I2C_MUX_H */
