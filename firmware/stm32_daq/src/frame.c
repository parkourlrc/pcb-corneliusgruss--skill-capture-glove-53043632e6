/**
 * frame.c — Sensor frame packing and timestamping
 *
 * Packs sensor data into the binary USB packet format.
 * All multi-byte fields are little-endian.
 */

#include "frame.h"
#include "config.h"
#include <string.h>

/* ── Little-endian packing helpers ────────────────────────────────────── */

static inline void pack_u16(uint8_t *buf, uint16_t val)
{
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)(val >> 8);
}

static inline void pack_u32(uint8_t *buf, uint32_t val)
{
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}

static inline void pack_i16(uint8_t *buf, int16_t val)
{
    pack_u16(buf, (uint16_t)val);
}

/* ── IMU Packet (Type 0x01, 17 bytes) ─────────────────────────────────── */

uint8_t frame_pack_imu(const imu_frame_t *imu, uint8_t *buf)
{
    buf[0] = PKT_TYPE_IMU;
    pack_u32(&buf[1], imu->timestamp_us);
    pack_i16(&buf[5],  imu->accel[0]);
    pack_i16(&buf[7],  imu->accel[1]);
    pack_i16(&buf[9],  imu->accel[2]);
    pack_i16(&buf[11], imu->gyro[0]);
    pack_i16(&buf[13], imu->gyro[1]);
    pack_i16(&buf[15], imu->gyro[2]);

    return PKT_SIZE_IMU;
}

/* ── Proprioceptive Packet (Type 0x02, 40 bytes) ─────────────────────── */

uint8_t frame_pack_proprio(
    uint32_t timestamp_us,
    const hall_data_t hall[4],
    const uint16_t tof[2],
    uint32_t sync_us,
    uint8_t status,
    uint16_t sync_seq,
    uint8_t *buf)
{
    /* [0] Packet type */
    buf[0] = PKT_TYPE_PROPRIO;

    /* [1-4] Timestamp */
    pack_u32(&buf[1], timestamp_us);

    /* [5-28] Hall sensor data: 4x (Bx, By, Bz) = 4x6 = 24 bytes */
    for (int i = 0; i < HALL_DATA_SLOTS; i++) {
        int off = 5 + i * 6;
        pack_i16(&buf[off + 0], hall[i].bx);
        pack_i16(&buf[off + 2], hall[i].by);
        pack_i16(&buf[off + 4], hall[i].bz);
    }

    /* [29-32] ToF distances: 2x uint16 = 4 bytes */
    pack_u16(&buf[29], tof[0]);
    pack_u16(&buf[31], tof[1]);

    /* [33-36] Sync pulse timestamp */
    pack_u32(&buf[33], sync_us);

    /* [37] Status flags */
    buf[37] = status;

    /* [38-39] Sync sequence counter */
    pack_u16(&buf[38], sync_seq);

    return PKT_SIZE_PROPRIO;  /* 40 */
}
