# RTL Implementation of Main Project

**Niklas Anderson**  
**Spring 2025**

## Overview

This challenge included creation of SystemVerilog (SV) modules corresponding to hardware functionality required based on the initial software/hardware boundary determined in Challenges 9 and 12. With the chosen boundary of the `step` function for iteration one, this involved creating modules for an `lstm_cell`, `matvec_mul`, `sigmoid_approx`, `tanh_approx`, `elementwise_add`, and `elementwise_mult`, in order to provide functionality for all of the calcuations performed within that function. In addition, a testbench was created in module `tb_lstm_cell`, which loaded in the weight data under the assumption that the final hardware implemlentation would allow for weight storage in SRAM.

The SV modules were created with assistance from ChatGPT, which resulted in challenges related to the LLM's inability to correctly generate code for SV in certain contexts, for example, ChatGPT often used blocking assignment inside `always_ff` blocks, which is illegal in SV. Though ChatGPT was directed to consider that synthesis was the ultimate goal, this iteration produced a module that was ultimately not synthesizable due to a variety of factors. These issues came to light in [Challenge 18](../challenge-18/README.md), and were addressed more in-depth in the [checkpoint document for Challenge 23](../challenge-23/README.md).

Beyond the synthesis issues, significant challenges were faced in generating the correct files for the weights, which were to be used in testing the SV implementation in simulation, and ultimately loaded into SRAM in the synthesized chip. This was due to several factors: one was that a layout for the weights in SRAM needed to be determine, which was non-trivial, especially given limited prior experience performing this work; a second factor was the limited familiarity with the actual structure of the weights, which made it difficult to verify both the requirements for the storage in SRAM, as well as the resulting files for correctness.

An attempt was made to generate golden vectors as well, to be used in functional verification. This effort suffered from some of the same limitations listed above and in previous challenges, in that the limited domain expertise made it difficult to understand and verify correctness of the generated golden vectors.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md).

## SystemVerilog Implementation

As described above, a vibe coding approach using ChatGPT was used to create the SV modules required for implementing the functionality for the calculations performed in the `step` Python function. ChatGPT required significant corrections at times, often related to non-blocking assignments and finite-state machine (FSM) logic.

## Weights and Golden Vector Generation

The weights and golden vector generation required the largest time expenditure for this challenge. Much of that work is visible in the directory for [Challenge 9](../challenge-09/), as that is where the saved model and weights file is located. An overview of the scripts created for this work is below.

### Commands

Generates .mem files for usage in SV simulation and testbench code:
```sh
python generate_mem_files.py --outdir weights
```
Produces weight files split up by model layer (e.g. decoders 1 or 2, encoders 1 or 2), gate (c, f, i, or o), and weight type (W, U, or b).

The following was used to generate golden vectors to be used in a testbench. **Note:** This code should be corrected as needed given the discovery of incorrect weight transposition.
```sh
python generate_golden_vectors.py
```

An attempt was made to verify correctness of the golden vectors, however, it was limited in that the verification primarily runs the same code as the inference script. So this suffers from the incorrect weight transposition noted above.
```sh
python validate_golden_vector.py --window 0
```

After the discovery of the weight transposition issue, scripts were updated as needed, and additional scripts were created to verify correctness of weight files.

The following checks the generated weight .mem files for correctness:
```sh
python verify_mem_file.py --npz saved_model/lstm_weights.npz --outdir weights
```

Additional functionality was consolidated in a `utils.py` file.

**Note:** Related to iteration two exploratory work, scripts such as `inspect_weights.py` and `minmax_vals.py` were created to better understand the weights and to determine the minimum and maximum values that were required to be represented. This informed the choice of fixed point formats.

## Results and Discussion

Due to the challenges described above related to weight file generation, attempts to synthesize the design were made in parallel to the work on weight file generation and functional verification. As such, fundamental issues with synthesis of the design were uncovered prior to completing functional verification. This provided to be beneficial, as less time was spent working towards completing the functionality of a design that had fundamental flaws. As a result, moving towards a second design iteration took place earlier than it would have if this process had moved in a strictly sequential order.

Despite the fact that this design had to be abandoned, significant learnings resulted from the attempted implementation. Specifically, a much better understanding of the model weights and how they might be used and represented in SV code was gained.
