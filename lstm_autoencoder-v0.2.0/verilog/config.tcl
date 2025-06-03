

# Floorplan defaults
# FP_CORE_UTIL may be increased to reduce cell density and
# make routing easier, but this will increase area.
# set ::env(FP_CORE_UTIL) 60
set ::env(FP_ASPECT_RATIO) 1.0
set ::env(FP_CORE_MARGIN) 2
# set ::env(PL_TARGET_DENSITY) 0.55

set ::env(SYNTH_MAX_FANOUT) 20

# Specify macro placement
# May try to let OpenLane try to handle this by commenting out the
# macro_placement.cfg file
# set ::env(MACRO_PLACEMENT_CFG) $::env(DESIGN_DIR)/macro_placement.cfg
set ::env(PL_MACROS_GRID) 1

# SRAM macro files
# set ::env(EXTRA_LEFS) "$::env(DESIGN_DIR)/macro/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8.lef"
# set ::env(EXTRA_GDS_FILES) "$::env(DESIGN_DIR)/macro/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8.gds"
# lappend ::env(LIB_FILES) "$::env(DESIGN_DIR)/macro/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib"

# Power/Ground nets for this design
# set ::env(VDD_PIN) "vccd1"
# set ::env(GND_PIN) "vssd1"

# Reducing power straps to attempt to reduce congestion and routing blockages
set ::env(FP_PDN_VPITCH) 40
set ::env(FP_PDN_HPITCH) 40

# Set max routing layer
# Might be able to increase this to met6 if the PDK supports it
set ::env(RT_MAX_LAYER) met5

# Enable congestion-aware placement
set ::env(PL_CONGESTION_EFFORT) high
