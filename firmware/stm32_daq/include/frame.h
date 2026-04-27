/**
 * frame.h — Sensor frame packing and timestamping
 *
 * Packs sensor data into the binary USB packet format.
 */

#ifndef FRAME_H
#define FRAME_H

#include <stdint.h>
#include "imu.h"
#include "hall_sensor.h"

/**
 * Pack an IMU frame into a 17-byte USB packet (type 0x01).
 * Returns the number of bytes written (always 17).
 */
uint8_t frame_pack_imu(const imu_frame_t *imu, uint8_t *buf);

/**
 * Pack a proprioceptive frame into a 40-byte USB packet (type 0x02).
 *
 * Layout (40 bytes):
 *   [0]     type (0x02)
 *   [1-4]   timestamp_us (uint32)
 *   [5-28]  hall 4x (Bx,By,Bz) = 24 bytes (int16)
 *   [29-32] tof 2x uint16 = 4 bytes
 *   [33-36] sync_us (uint32)
 *   [37]    status flags (uint8)
 *   [38-39] sync_seq (uint16)
 */
uint8_t frame_pack_proprio(
    uint32_t timestamp_us,
    const hall_data_t hall[4],
    const uint16_t tof[2],
    uint32_t sync_us,
    uint8_t status,
    uint16_t sync_seq,
    uint8_t *buf
);

#endif /* FRAME_H */
