cmake_minimum_required(VERSION 3.20)

# ── Toolchain ──────────────────────────────────────────────────────────
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_SIZE arm-none-eabi-size)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(glove_daq C ASM)

# ── MCU flags ──────────────────────────────────────────────────────────
set(MCU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS "${MCU_FLAGS} -Wall -Wextra -fdata-sections -ffunction-sections -fno-common")
set(CMAKE_C_FLAGS_DEBUG "-Og -g3 -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# ── STM32 HAL paths (adjust to your local install) ────────────────────
# Download from: https://github.com/STMicroelectronics/STM32CubeG4
set(STM32_HAL_DIR "${CMAKE_SOURCE_DIR}/lib/STM32CubeG4" CACHE PATH "Path to STM32CubeG4 SDK")
set(CMSIS_DIR "${STM32_HAL_DIR}/Drivers/CMSIS")
set(HAL_DIR "${STM32_HAL_DIR}/Drivers/STM32G4xx_HAL_Driver")

# ── TinyUSB path ──────────────────────────────────────────────────────
set(TINYUSB_DIR "${CMAKE_SOURCE_DIR}/lib/tinyusb" CACHE PATH "Path to TinyUSB")

# ── Sources ────────────────────────────────────────────────────────────
# For blink test: only blink_test.c + system_clock.c
# For full firmware: use file(GLOB APP_SOURCES "src/*.c") and remove blink_test.c
option(BLINK_TEST "Build minimal blink test instead of full firmware" OFF)

if(BLINK_TEST)
    set(APP_SOURCES
        src/blink_test.c
        src/system_clock.c
        src/usb_descriptors.c
    )
else()
    file(GLOB APP_SOURCES "src/*.c")
    list(FILTER APP_SOURCES EXCLUDE REGEX "blink_test\\.c$")
endif()

set(HAL_SOURCES
    ${HAL_DIR}/Src/stm32g4xx_hal.c
    ${HAL_DIR}/Src/stm32g4xx_hal_cortex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_rcc.c
    ${HAL_DIR}/Src/stm32g4xx_hal_rcc_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_pwr.c
    ${HAL_DIR}/Src/stm32g4xx_hal_pwr_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_gpio.c
    ${HAL_DIR}/Src/stm32g4xx_hal_spi.c
    ${HAL_DIR}/Src/stm32g4xx_hal_spi_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_i2c.c
    ${HAL_DIR}/Src/stm32g4xx_hal_i2c_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_tim.c
    ${HAL_DIR}/Src/stm32g4xx_hal_tim_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_dma.c
    ${HAL_DIR}/Src/stm32g4xx_hal_dma_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_flash.c
    ${HAL_DIR}/Src/stm32g4xx_hal_flash_ex.c
    ${HAL_DIR}/Src/stm32g4xx_hal_flash_ramfunc.c
    ${CMSIS_DIR}/Device/ST/STM32G4xx/Source/Templates/system_stm32g4xx.c
    ${CMSIS_DIR}/Device/ST/STM32G4xx/Source/Templates/gcc/startup_stm32g431xx.s
)

set(TINYUSB_SOURCES
    ${TINYUSB_DIR}/src/tusb.c
    ${TINYUSB_DIR}/src/common/tusb_fifo.c
    ${TINYUSB_DIR}/src/device/usbd.c
    ${TINYUSB_DIR}/src/device/usbd_control.c
    ${TINYUSB_DIR}/src/class/cdc/cdc_device.c
    ${TINYUSB_DIR}/src/portable/st/stm32_fsdev/dcd_stm32_fsdev.c
    ${TINYUSB_DIR}/src/portable/st/stm32_fsdev/fsdev_common.c
)

# ── Executable ─────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME}.elf
    ${APP_SOURCES}
    ${HAL_SOURCES}
    ${TINYUSB_SOURCES}
)

# ── Include paths ──────────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME}.elf PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMSIS_DIR}/Include
    ${CMSIS_DIR}/Device/ST/STM32G4xx/Include
    ${HAL_DIR}/Inc
    ${TINYUSB_DIR}/src
)

# ── Preprocessor defines ──────────────────────────────────────────────
target_compile_definitions(${PROJECT_NAME}.elf PRIVATE
    STM32G431xx
    USE_HAL_DRIVER
)

# ── Linker ─────────────────────────────────────────────────────────────
set(LINKER_SCRIPT "${CMAKE_SOURCE_DIR}/linker/STM32G431CBTx.ld")

target_link_options(${PROJECT_NAME}.elf PRIVATE
    -T${LINKER_SCRIPT}
    -mcpu=cortex-m4
    -mthumb
    -mfpu=fpv4-sp-d16
    -mfloat-abi=hard
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    --specs=nano.specs
    --specs=nosys.specs
    -lc -lm -lnosys
)

# ── Post-build: generate .bin and .hex ─────────────────────────────────
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>
    COMMENT "Generating .bin and .hex, printing size"
)
