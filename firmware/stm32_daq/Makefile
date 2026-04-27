# Convenience wrapper for CMake build

BUILD_DIR ?= build
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Debug

.PHONY: all clean flash debug

all:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) .. && make -j$$(nproc)

release:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc)

clean:
	@rm -rf $(BUILD_DIR)

flash: all
	openocd -f openocd.cfg -c "program $(BUILD_DIR)/glove_daq.bin verify reset exit 0x08000000"

debug:
	openocd -f openocd.cfg &
	arm-none-eabi-gdb $(BUILD_DIR)/glove_daq.elf \
		-ex "target remote localhost:3333" \
		-ex "monitor reset halt" \
		-ex "load"

size: all
	arm-none-eabi-size $(BUILD_DIR)/glove_daq.elf
