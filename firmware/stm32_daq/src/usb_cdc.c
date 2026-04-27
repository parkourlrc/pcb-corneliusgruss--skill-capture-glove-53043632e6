/**
 * usb_cdc.c — TinyUSB CDC device with ring buffer TX
 *
 * Uses TinyUSB's device stack for USB CDC-ACM. Data is queued into a
 * ring buffer from ISR/main context and drained to TinyUSB in the main loop.
 */

#include "usb_cdc.h"
#include "config.h"
#include "tusb.h"

/* ── Ring buffer ──────────────────────────────────────────────────────── */

static uint8_t tx_buf[USB_TX_BUF_SIZE];
static volatile uint32_t tx_head = 0;
static volatile uint32_t tx_tail = 0;

static uint32_t ring_free(void)
{
    uint32_t h = tx_head;
    uint32_t t = tx_tail;
    if (h >= t)
        return USB_TX_BUF_SIZE - 1 - (h - t);
    return t - h - 1;
}

/* ── Public API ───────────────────────────────────────────────────────── */

void usb_cdc_init(void)
{
    /* STM32G4: disable UCPD dead battery pull-downs on USB D+/D- */
    __HAL_RCC_PWR_CLK_ENABLE();
    PWR->CR3 |= PWR_CR3_UCPD_DBDIS;

    __HAL_RCC_USB_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* Set priority for all three USB IRQs before tusb_init() enables them */
    HAL_NVIC_SetPriority(USB_HP_IRQn, 3, 0);
    HAL_NVIC_SetPriority(USB_LP_IRQn, 3, 0);
    HAL_NVIC_SetPriority(USBWakeUp_IRQn, 3, 0);

    /* tusb_init() enables the IRQs internally via fsdev_int_enable() */
    tusb_init();
}

uint32_t usb_cdc_write(const uint8_t *data, uint32_t len)
{
    uint32_t free = ring_free();
    if (len > free)
        len = free;

    for (uint32_t i = 0; i < len; i++) {
        tx_buf[tx_head] = data[i];
        tx_head = (tx_head + 1) % USB_TX_BUF_SIZE;
    }
    return len;
}

void usb_cdc_flush(void)
{
    if (!tud_cdc_connected())
        return;

    while (tx_tail != tx_head) {
        uint32_t avail = tud_cdc_write_available();
        if (avail == 0)
            break;

        /* Write contiguous chunk from tail to end-of-buffer or head */
        uint32_t h = tx_head;
        uint32_t t = tx_tail;
        uint32_t count;

        if (h > t) {
            count = h - t;
        } else {
            count = USB_TX_BUF_SIZE - t;
        }
        if (count > avail)
            count = avail;

        uint32_t written = tud_cdc_write(&tx_buf[t], count);
        tx_tail = (tx_tail + written) % USB_TX_BUF_SIZE;
    }

    tud_cdc_write_flush();
}

void usb_cdc_task(void)
{
    tud_task();
}

bool usb_cdc_connected(void)
{
    return tud_cdc_connected();
}

/* ── USB Interrupt Handlers — G4 has three USB IRQs ──────────────────── */

void USB_HP_IRQHandler(void)
{
    tud_int_handler(0);
}

void USB_LP_IRQHandler(void)
{
    tud_int_handler(0);
}

void USBWakeUp_IRQHandler(void)
{
    tud_int_handler(0);
}

/* ── TinyUSB Callbacks ────────────────────────────────────────────────── */

void tud_mount_cb(void) {}
void tud_umount_cb(void) {}
void tud_suspend_cb(bool remote_wakeup_en) { (void)remote_wakeup_en; }
void tud_resume_cb(void) {}

void tud_cdc_line_state_cb(uint8_t itf, bool dtr, bool rts)
{
    (void)itf;
    (void)dtr;
    (void)rts;
}

void tud_cdc_rx_cb(uint8_t itf)
{
    (void)itf;
    /* Host-to-device commands not implemented yet */
}
