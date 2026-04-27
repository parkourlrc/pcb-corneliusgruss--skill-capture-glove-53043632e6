/**
 * usb_cdc.h — TinyUSB CDC device with DMA TX ring buffer
 */

#ifndef USB_CDC_H
#define USB_CDC_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Initialize TinyUSB CDC device. Must be called after clock and CRS setup.
 */
void usb_cdc_init(void);

/**
 * Queue data for USB transmission. Non-blocking — copies into ring buffer.
 * Returns the number of bytes actually queued (may be less than len if full).
 */
uint32_t usb_cdc_write(const uint8_t *data, uint32_t len);

/**
 * Flush the TX ring buffer to TinyUSB. Call from main loop.
 */
void usb_cdc_flush(void);

/**
 * TinyUSB task — must be called from main loop to service USB events.
 */
void usb_cdc_task(void);

/**
 * Check if USB is connected and CDC port is open.
 */
bool usb_cdc_connected(void);

#endif /* USB_CDC_H */
