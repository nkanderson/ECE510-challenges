# NOTE: This does not work. Saving it for reference in case another
# attempt is made with openram in the future.
import openram

# openram.init_openram("mem_to_rom_data.py")

from openram import rom

r = rom("mem_to_rom_data.py")

# Output the files for the resulting ROM
r.save()

# Delete temp files etc.
openram.end_openram()
