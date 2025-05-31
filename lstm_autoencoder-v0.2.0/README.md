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

In order to consider alternate design options, a smaller mac2 variant was created and synthesized using OpenLane2. Clock speed improvements were marginal, with a minimum period somewhere around 30ns. It seems unlikely this approach is worth pursuing, but it may be worth considering if many mac instances may be used in parallel in the matvec_mul module.

## Future Work

### Controller for Multiple Layers, Gates, and Weights

In order to specify the specific weights to use for a given matrix-vector multiplication option, we will need to allow for a control word to be passed, along with the vector input. The control word should have enough bits to select the appropriate layer (4 options), gate (4 options), and weight (`W` or `U`), as well as one additional option to select the dense layer. This will require 6 bits in order to cover 32 combinations for the main encoder and decoder layers and their gates and weights, plus 1 for the dense layer, or 33 total. There will essentially be a "dense" bit.

Another alternative is to use a basic FSM to select the appriopriate combination from the above in the order called by the algorithm. This may be the simplest next step before implementing a control word. The FSM could be implemented within the SRAM memory controller for loading the weights.

### SRAM Addressing

One possible option would be to organize all matrices (`W` and `U`, may include `b` vector here as well for future usage) for all layers and gates in a single flat memory using multiple SRAM macros to create logical banks. For simplicity, it may be easiest to initially base each layer size on the largest required layer size, which should be one of the layers with a hidden size of 64.

The memory address offset for a given layer, gate, and weight will be calculated in the following manner:
```
offset = BASE 
       + LAYER_ID * LAYER_STRIDE 
       + GATE_ID * GATE_STRIDE 
       + MATRIX_ID * MATRIX_STRIDE;
```

LAYER_STRIDE = Max total layer size = space for 4 gates × 3 matrices (W, U, b) = 12 matrix slots of varying size
GATE_STRIDE = 3 matrices - TODO: It may be possible to have a base value and scale it depending on the particular layer
MATRIX_STRIDE = rows × cols - Determined by the client code

This is not the most space-efficient, but could be a relatively simple inital organization.
