/**
 * button.h — SW_RECORD (PB3) with debounce
 */

#ifndef BUTTON_H
#define BUTTON_H

#include <stdbool.h>

/**
 * Initialize PB3 as input with internal pull-up.
 */
void button_init(void);

/**
 * Poll the button state with software debounce (~20ms).
 * Call this from the main loop at ≥50Hz.
 */
void button_update(void);

/**
 * Returns true if the button is currently held down (debounced).
 */
bool button_is_pressed(void);

/**
 * Returns true once per press event (rising edge, debounced).
 * Cleared after reading.
 */
bool button_was_pressed(void);

#endif /* BUTTON_H */
