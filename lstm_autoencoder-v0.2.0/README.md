# LSTM Autoencoder Hardware Acceleration v0.2.0

## Overview

This work represents the iterative changes outlined in [Challenge 23](../challenge-23/README.md).

## Summary of Work

### Updated Profiling

After fixing the weight loading transposition issue (see [CHANGELOG](./docs/CHANGELOG.md)), the overall runtime for inference was much longer, going from ~2s to ~60s. The identified bottlenecks were the same, however, so at this point, it does not seem like changing hardware acceleration targets is necessary. It may be worthwhile to profile on multiple machines and attempt to confirm the updated runtime though, as it will impact the ultimate assessment of hardware execution speedup.

### Reduced Precision Values

With assistance from ChatGPT for analysis of existing inference and weights, determined the following fixed point formats should suffice:

| Signal Type      | Max abs | Recommended Q-format | Notes                                  |
| ---------------- | ------- | -------------------- | -------------------------------------- |
| `matvec_mul`     | ±3.3    | `Q4.12`              | Range: ±8.0, resolution ≈ 0.00024      |
| `sigmoid input`  | ±4.2    | `Q4.12`              | Same                                   |
| `sigmoid output` | \[0, 1] | `U1.15`              | Unsigned                               |
| `tanh input`     | ±8      | `Q5.11` or `Q6.10`   | Slightly wider range needed            |
| `tanh output`    | ±1      | `Q2.14`              | Very precise near 0; saturates near ±1 |

Weights: `Q2.14`
Biases: also `Q2.14`

Since our matrix-vector multiply consistently works on weights (either `W` or `U`) and an input vector (which should have the format `Q4.12`), it should be made to work specifically with those formats. ChatGPT suggests the following approach:

1. Multiply one weight element (Q2.14) × one vector element (Q4.12):
  - Total fractional bits: 14 + 12 = 26
  - Bit width: 16b × 16b → 32b product
  - Result format: Q6.26

2. Sum across dot product:
  - You accumulate many Q6.26 values
  - Accumulator might be, say, 40–48 bits to preserve full precision

3. Truncate/round the final sum to Q4.12 (or Q5.11, Q6.10 depending on dynamic range)

### Time-Multilplexed Matrix-Vector Multiplier in SV

TODO:
- Will want to flatten the 2D input ports prior to synthesis. It should have flattened wires or packed arrays for all inputs and outputs.
- In simulation, the shared memory (sram) can be a logic signed [15:0] sram[0:4095] array instantiated in the testbench and passed by reference to both matrix_loader and matvec_tm_multiplier.
  - In synthesis, we'll eventually replace this with a dual-port SRAM macro (e.g., sky130_sram_1rw1r_32x1024_8) and mux the address/data paths between reader and writer. (Though I'm not sure we'll need a writer, so this part may be simplified.)

### OpenLane2 Synthesis

Synthesized the mac4 SystemVerilog by first generating verilog code using the sv2v CLI tool, then creating a minimal `config.json` file. Results from initial synthesis indicate timing closure was achieved with a clock period of 40ns for this combinational circuit, resulting in a max clock frequency of 25MHz. In intepreting the `metrics.json output`, ChatGPT noted the critical path was around 26ns. It seems like it would be possible to optimize the design to get closer to a 26ns clock period, but may take significant time configuring settings.

ChatGPT noted the following metrics to review from the `metrics.json` in order to interpret timing data:
- `timing__setup_ws` - worst-case setup path delay, which should be close to the actual critical path
- Timing violations:
  - `timing__setup_vio__count`
  - `timing__hold_vio__count`
  - `timing__setup_wns`
  - `timing__tns`

There may also be design rule violations, related to max slew constraints.
