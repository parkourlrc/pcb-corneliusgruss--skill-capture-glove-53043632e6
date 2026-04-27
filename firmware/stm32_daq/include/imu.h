/**
 * imu.h — QMI8658C IMU driver (bit-bang I2C on SPI pins)
 *
 * The board was designed for ICM-42688-P on SPI1, but JLCPCB placed a
 * QMI8658C clone (Tokmas ICM-42688-PC). The QMI8658C uses I2C at 0x6A.
 * We bit-bang I2C on PA5(SCL)/PA7(SDA) with PA4(CS) held high.
 *
 * Call imu_poll() from the main loop — it reads at IMU_RATE_HZ.
 * Frames are pushed to a ring buffer, same interface as before.
 */

#ifndef IMU_H
#define IMU_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint32_t timestamp_us;
    int16_t accel[3];  /* Raw 16-bit, ±2g: LSB ≈ 0.061 mg */
    int16_t gyro[3];   /* Raw 16-bit, ±256 dps: LSB ≈ 7.8 mdps */
} imu_frame_t;

/**
 * Initialize QMI8658C via bit-bang I2C. Returns true if WHO_AM_I = 0x05.
 */
bool imu_init(void);

/**
 * Poll the IMU — call from main loop. Reads at IMU_RATE_HZ internally.
 */
void imu_poll(void);

/**
 * Check if a new IMU frame is available in the ring buffer.
 */
bool imu_available(void);

/**
 * Pop the oldest frame from the ring buffer.
 */
bool imu_read(imu_frame_t *frame);

#endif /* IMU_H */
