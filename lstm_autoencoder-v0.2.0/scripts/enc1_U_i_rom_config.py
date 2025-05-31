# OpenRAM configuration for enc1_U_i ROM
output_path = "generated"
output_name = "enc1_U_i_rom"

word_size = 16  # Q2.14 values = 16 bits per word
num_words = 4096  # 64x64 weights
num_rw_ports = 0
num_r_ports = 1
num_w_ports = 0

# Optional: ensure deterministic physical layout
tech_name = "sky130"
process_corners = ["TT"]

# Clock is needed for ROM access
route_supplies = True
check_lvsdrc = True
perimeter_pins = True

# ROM data file (single line of 16384 hex characters)
rom_data_file = "enc1_U_i.data"
data_type = "hex"  # Data type for the ROM

# Keep analytical placement simple
recompute_sizes = False
trim_netlist = False
