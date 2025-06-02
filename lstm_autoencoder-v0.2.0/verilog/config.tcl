set ::env(DESIGN_NAME) matvec_multiplier

set ::env(CLOCK_PORT) "clk"
set ::env(CLOCK_PERIOD) "40.0"

# Include all source files
set ::env(VERILOG_FILES) [glob $::env(DESIGN_DIR)/src/*.v]

# Floorplan defaults
set ::env(FP_CORE_UTIL) 40
set ::env(FP_ASPECT_RATIO) 1.0
set ::env(FP_CORE_MARGIN) 2

# Allow automatic macro placement
# Going to let OpenLane try to handle this initially
# set ::env(MACRO_PLACEMENT_CFG) $::env(DESIGN_DIR)/macro_placement.cfg
set ::env(PL_MACROS_GRID) 1

# SRAM macro files
set ::env(EXTRA_LEFS) "$::env(DESIGN_DIR)/macro/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8.lef"
set ::env(EXTRA_GDS_FILES) "$::env(DESIGN_DIR)/macro/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8.gds"
lappend ::env(LIB_FILES) "$::env(DESIGN_DIR)/macro/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib"

# Power/Ground nets for this design
set ::env(VDD_PIN) "vccd1"
set ::env(GND_PIN) "vssd1"
