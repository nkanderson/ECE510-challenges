import cocotb
from cocotb.triggers import Timer


def to_fixed(val, total_bits=16, frac_bits=12):
    """Convert float to Qm.n fixed-point format (default Q4.12) as signed int."""
    scaled = int(round(val * (1 << frac_bits)))
    if scaled < 0:
        scaled += 1 << total_bits
    return scaled & ((1 << total_bits) - 1)


def to_signed(val, bits):
    """Convert val to signed integer with 'bits' width."""
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val

def mac4_model(a_vals, b_vals):
    products = []
    for a, b in zip(a_vals, b_vals):
        a_fixed = to_signed(to_fixed(a, total_bits=16, frac_bits=14), 16)  # Q4.12
        b_fixed = to_signed(to_fixed(b, total_bits=16, frac_bits=12), 16)  # Q2.14
        product = a_fixed * b_fixed      # Q6.26
        # print(f"product:{product} (a_fixed=0x{a_fixed:04X}): b_fixed=0x{b_fixed:04X}")
        shifted = product >> 14          # Q6.12 (>>> in SV is arithmetic shift)
        # print(f"shifted:{shifted} (a_fixed=0x{a_fixed:04X}): b_fixed=0x{b_fixed:04X}")
        products.append(shifted)
    return sum(products)



@cocotb.test()
async def test_mac4_basic(dut):
    """Test mac4 with simple known inputs."""
    # Example inputs
    # a_vals are in Q2.14 format
    a_vals = [0.5, -1.0, 0.25, 1.0]
    # b_vals are in Q4.12 format
    b_vals = [1.0, 0.5, -0.5, -0.25]
    # a_vals = [0.5, -1.5, 1.0, 0.25]
    # b_vals = [4.0, 1.0, -2.25, 4.0]
    # a_vals = [1.5, 1.5, 1.5, 1.5]
    # b_vals = [4.0, 4.0, 4.0, 4.0]

    # Apply fixed-point inputs
    for i, val in enumerate(a_vals):
        fixed_val = to_fixed(val, total_bits=16, frac_bits=14)
        setattr(dut, f"a{i}", fixed_val)
        dut._log.info(f"a{i} (float={a_vals[i]}): fixed=0x{to_fixed(val):04X}")

    for i, val in enumerate(b_vals):
        fixed_val = to_fixed(val, total_bits=16, frac_bits=12)
        setattr(dut, f"b{i}", fixed_val)
        dut._log.info(f"b{i} (float={b_vals[i]}): fixed=0x{to_fixed(val):04X}")

    await Timer(1, units="ns")

    # Compute expected result in Q4.12
    expected_sum = mac4_model(a_vals, b_vals)

    # Read DUT output and interpret as signed 32-bit integer
    result = int(dut.result.value)
    dut._log.info(f"Result: {result}")
    result_signed = to_signed(result, 32)

    # Compare result
    dut._log.info(f"Expected: {expected_sum}, DUT output: {result_signed}")
    assert result_signed == expected_sum, f"Mismatch!"

