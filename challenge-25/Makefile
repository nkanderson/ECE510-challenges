# Makefile

# defaults
SIM ?= verilator
# Use verilog value for verilog or systemverilog
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(PWD)/mac4.sv

# TOPLEVEL is the name of the toplevel module in Verilog or VHDL file
TOPLEVEL = mac4

# MODULE is the basename of the Python test file
MODULE = test_mac4

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
