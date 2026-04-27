/**
 * tusb_config.h — TinyUSB configuration for Glove DAQ
 */

#ifndef TUSB_CONFIG_H
#define TUSB_CONFIG_H

/* ── MCU port ─────────────────────────────────────────────────────────── */
#define CFG_TUSB_MCU                OPT_MCU_STM32G4

/* ── Role ─────────────────────────────────────────────────────────────── */
#define CFG_TUSB_RHPORT0_MODE       OPT_MODE_DEVICE
#define CFG_TUD_ENABLED             1
#define CFG_TUH_ENABLED             0

/* ── OS ───────────────────────────────────────────────────────────────── */
#define CFG_TUSB_OS                 OPT_OS_NONE

/* ── Memory ───────────────────────────────────────────────────────────── */
#define CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_ALIGN          __attribute__((aligned(4)))

/* ── Device configuration ─────────────────────────────────────────────── */
#define CFG_TUD_ENDPOINT0_SIZE      64

/* ── Class drivers ────────────────────────────────────────────────────── */
#define CFG_TUD_CDC                 1
#define CFG_TUD_MSC                 0
#define CFG_TUD_HID                 0
#define CFG_TUD_MIDI                0
#define CFG_TUD_VENDOR              0

/* ── CDC FIFO sizes ───────────────────────────────────────────────────── */
#define CFG_TUD_CDC_RX_BUFSIZE      256
#define CFG_TUD_CDC_TX_BUFSIZE      2048

#endif /* TUSB_CONFIG_H */
