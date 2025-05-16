# LSTM Autoencoder Hardware Acceleration v0.2.0

## Overview

This work represents the iterative changes outlined in [Challenge 23](../challenge-23/README.md).

## Summary of Work

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
