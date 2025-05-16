# Week 7 Checkpoint

This checkpoint serves to document current status of the project as it relates to the definition of project success described in [w7_codefest.pdf](./w7_codefest.pdf), as well as outline next steps to achieve improvement in the next iteration. At a high level, all stages of the project have been attempted, though with varying success. Specifically, challenges have come up in functional verification of the model in SV, and the transistor-level design using OpenLane2 has not successfully completed. However, a better understanding of the requirements for each stage has been gained. The first iteration may be described as a relatively naive LLM-driven attempt, but it has been useful in gaining a better understanding of the project at all levels.

The discussion below includes high level changes to consider for the next iteration, specific possible next steps, technical details related to the candidate changes, and finally a concrete list of the decided-upon next steps to execute.

## Current Status

- Processed a NOAA dataset to prepare it for consumption by the training model.
- Created an LSTM Autoencoder model by training on NOAA data using 6 features from the dataset. The model was saved in the form of a numpy npz file `lstm_weights.npz`.
- Created a pure Python inference model which performs matrix multiplication and elementwise addition and multiplication using sequential operations.
- To check model performance, inference was run on a new dataset using the original weights.
- Profiled the inference algorithm and determined the `step` method is a bottleneck, and at a level below that, the matrix multiplication is another bottleneck. The matrix multiplication is performed more frequently by an order of magnitude.
- Chose an initial HW / SW boundary at the `step` level, as this method represented a significant proportion of overall execution time and encapsulated multiple matrix multiplication calls.
- Decided to try to put the weights in SRAM in the final hardware design.
- Calculated approximate communication cost in terms of data transfer between software and hardware and determined a maximum target execution time for the hardware implementation of the `step` method in order for it to improve on software execution time.
- Created the following modules in SystemVerilog: `lstm_cell`, `matvec_mul`, `sigmoid_approx`, `tanh_approx`, `elementwise_add`, and `elementwise_mul`.
- Created scripts to export weights from original model into a format that could be used in SystemVerilog. **Note**: This step by itself proved more time-consuming than initially expected. In particular, verifying correctness and determining a sensible form for the data in SV to translate to SRAM was challenging.
- Created scripts to export golden vectors in order to perform functional verification on SV model with the `tb_lstm_cell` testbench. This was also challenging, and functional verification of the `lstm_cell` as a whole has not been completed.
- In parallel with the above work on SV module created and verification, synthesis was attempted using OpenLane2 with an early version which compiled and ran successfully (though again, correctness was not yet verified). This was not successful either, as the OpenLane2 series of operations took an excessive amount of time to execute. 57 out of the 78 stages completed, and an lef file was manually generated for stage 58. However, the excessive amount of time for execution is considered a red flag for the design as a whole.

### Autoencoder Model Detail

The current autoencoder model is composed of 2 encoder and 2 decoder LSTM layers. Encoder 1 has a hidden size of 64, encoder 2 a hidden size of 32. Decoder 1 has a hidden size of 32, decoder 2 a hidden size of 64. There is a repeater layer in between the decoders and encoders, and a dense layer for a final linear transformation.

Six features were included from the NOAA dataset: air temperature, sea surface temperature, sea level pressure, wind speed, wave height, and wave period.

The inference algorithm is composed of 2 encoder and 2 decoder LSTM layers, and a final dense layer.

## Overview of Changes in Next Iteration

### Try a different SW / HW boundary at lower level

It may make sense to begin hardware acceleration at the matrix multiplication level, or perhaps at the level of computation for each gate within the LSTM Cell `step`. Each gate computation is composed of an activation function, multiple elementwise operations, and multiple matrix multiplication operations.

### Consider simplifying model

Reducing the number of layers from 2 each for encoders and decoders to 1 each would significantly reduce the overall size of the weights. Within each cell, reducing the number of hidden neurons from 64 to 32 would reduce the weight for an individual gate.

Beyond that, reducing the number of features from 6 to 2 would reduce the overall sizing.

### Reduce precision in hardware

The current model uses 32-bit precision. It could be worth attempting to use 16-bit precision in order to reduce the input and output wires required for the modules at the hardware level.

## Possible Next Steps

- Recreate a simplified model with 1 encoder and 1 decoder, each with 32 hidden cells. Though 64 hidden cells may be ok, depending on the outcome of the matrix multiplication module work.
- Reduce features in the model from 6 to 2: air temperature and sea level pressure.
- Recreate inference algorithm as needed with new model. Need to determine the sizing of the input and output vectors. Benchmark new algorithm.
- Output weights and golden vectors for the new model.
- Estimate the required input and output sizes necessary in hardware for matrix multiplication and decide on an implementation that will be feasible to synthesize. Consider creating a small version and running through OpenLane before expanding.
- Consider reducing precision for hardware, starting with the matrix multiplication, and impact of this change.
- Might try to get weights into SRAM with the above matrix multiplier, but may also just need to start by assuming they will be transferred per operation. If attempting to use SRAM, will need to use SRAM macros for Sky130 tech. Per ChatGPT, it would be advisable to split the stored weights across multiple banks for blocks to allow for parallel access and help with timing closure and routing. See below for technical details.
- Consider expanding hardware implementation to cover each gate computation - this would include the activation function applied to elementwise add on the result of matrix multiplication on the input kernel and the input with result of elementwise add of the matrix multiplication of the recurrent kernel and the previous hidden state with the bias (see `lstm_core.py` `step` method).

## Technical Details

### SRAM

Per ChatGPT, to use SRAM macros in an ASIC design:

Can use behavioral memory in early RLT, for example:
```systemverilog
module memory (
    input logic clk,
    input logic we,
    input logic [9:0] addr,
    input logic [31:0] din,
    output logic [31:0] dout
);
    logic [31:0] mem_array [0:1023];

    always_ff @(posedge clk) begin
        if (we)
            mem_array[addr] <= din;
        dout <= mem_array[addr];
    end
endmodule
```
The above will map to flip-flops or distributed RAM, but works for early-stage simulation and functional verification.

Later in the process, or as an alternative to just the above in simulation, abstract with a memory wrapper:
```systemverilog
module memory_wrapper (
    input logic clk,
    input logic we,
    input logic [9:0] addr,
    input logic [31:0] din,
    output logic [31:0] dout
`ifdef USE_SRAM_MACRO
    // synthesis-specific ports if needed
`endif
);

`ifdef USE_SRAM_MACRO
    // Instantiate real SRAM macro for synthesis
    sky130_sram_1rw1r_32x1024_8 sram_inst (
        .clk0(clk),
        .csb0(~1'b1),  // chip select active-low
        .web0(~we),    // write enable active-low
        .addr0(addr),
        .din0(din),
        .dout0(dout)
    );
`else
    // Behavioral RAM for simulation
    logic [31:0] mem_array [0:1023];

    always_ff @(posedge clk) begin
        if (we)
            mem_array[addr] <= din;
        dout <= mem_array[addr];
    end
`endif

endmodule
```

During simulation, ensure `USE_SRAM_MACRO` is not defined, and during synthesis, makre sure it is defined within the toolchain.

During synthesis, need to point to `.lib`, `.lef`, `.gds` of the SRAM macro.

**Note for personal reference**: [ChatGPT conversation on SRAM macros](https://chatgpt.com/c/682387eb-d5ec-8012-a494-21e332a51da6).

### Tensor Shapes

Shapes for first iteration model. Tensor shapes determined with ChatGPT assistance.

#### LSTM Cells

| Step                     | Input Shape(s)     | Output Shape |
| ------------------------ | ------------------ | ------------ |
| `lstm1.step(x, h1, c1)`  | `[6], [64], [64]`  | `[64], [64]` |
| `lstm2.step(h1, h2, c2)` | `[64], [32], [32]` | `[32], [32]` |
| `repeated = [h2 × 24]`   | `[32]`             | `[24, 32]`   |
| `lstm3.step(x, h3, c3)`  | `[32], [32], [32]` | `[32], [32]` |
| `lstm4.step(h3, h4, c4)` | `[32], [64], [64]` | `[64], [64]` |
| `dense.forward(h4)`      | `[64]`             | `[6]`        |


#### Gates inside cells

| Function                  | Input Shapes                                                | Output Shape    |
| ------------------------- | ----------------------------------------------------------- | --------------- |
| `matvec_mul(W_i, x)`      | `W_i: [hidden_size × input_size]`, `x: [input_size]`        | `[hidden_size]` |
| `matvec_mul(U_i, h_prev)` | `U_i: [hidden_size × hidden_size]`, `h_prev: [hidden_size]` | `[hidden_size]` |
| `elementwise_add(...)`    | `[hidden_size], b_i: [hidden_size]`                         | `[hidden_size]` |
| `vector_sigmoid(...)`     | `[hidden_size]`                                             | `[hidden_size]` |

The above shows variables for gate `i` specifically, but uses variable terms like `hidden_size` which should apply to all gates. So for LSTM cell 1 (encoder 1), the kernel weight `W` for a single gate is the hidden size of 64 times the input size of 6. The recurrent weight *for a single gate* is hidden size of 64 times hidden size of 64, meaning it is made up of 4096 float values. For all gates for a cell, that value is multiplied times 4.

#### Final cell and hidden states

New cell state

| Function               | Input Shapes                   | Output Shape    |
| ---------------------- | ------------------------------ | --------------- |
| `elementwise_mul(...)` | `[hidden_size], [hidden_size]` | `[hidden_size]` |
| `elementwise_add(...)` | `[hidden_size], [hidden_size]` | `[hidden_size]` |


New hidden state

| Function               | Input Shapes                   | Output Shape    |
| ---------------------- | ------------------------------ | --------------- |
| `vector_tanh(c)`       | `[hidden_size]`                | `[hidden_size]` |
| `elementwise_mul(...)` | `[hidden_size], [hidden_size]` | `[hidden_size]` |

#### Summary for gates and final states

| Function                | Input Types & Shapes               | Output Shape    | Type    |
| ----------------------- | ---------------------------------- | --------------- | ------- |
| `matvec_mul(W, x)`      | `List[List[float]]`, `List[float]` | `[hidden_size]` | `float` |
| `matvec_mul(U, h_prev)` | `List[List[float]]`, `List[float]` | `[hidden_size]` | `float` |
| `elementwise_add(a, b)` | `List[float], List[float]`         | `[hidden_size]` | `float` |
| `vector_sigmoid(vec)`   | `List[float]`                      | `[hidden_size]` | `float` |
| `vector_tanh(vec)`      | `List[float]`                      | `[hidden_size]` | `float` |
| `elementwise_mul(a, b)` | `List[float], List[float]`         | `[hidden_size]` | `float` |

**Note for personal reference**: [ChatGPT conversation on tensor shapes in model](https://chatgpt.com/c/682370ab-7954-8012-93dd-e9b11bef77c8).

### Input / Output Wires

The number of hidden neurons determines the input size for encoder 2 and output size for decoder 2. The number of hidden neurons also determines the overall size of the weights, since weight for the input kernel is determined by hidden size * 4 (for number of gates) * input size, and for the recurrent kernel hidden size * 4 * hidden size. This is relevant to the above SRAM discussion, but informs the decision to abandon passing weights through inputs and outputs given the large number of wires and large amount of data transfer that would be required.

### Matrix Multiplication Analysis

Given the above, the maximum size inputs and outputs for matrix multiplication if we assume the weights can be stored in SRAM rather than transfered would be determined by largest hidden size for a cell. This is the case, at least, for an implementation that is not using streaming or time-multiplexing, or a tiled systolic array.

## Iteration Two Plan

- Determine max and min values that will need to be represented in hardware to determine the fixed point format. Planning on using a 16-bit representation.
- Begin with a time-multiplexed matrix-vector multiplication unit that handles a max vector size of 16 values at a time. Use dummy weights for the time being, but attempt to use the matrix sizes corresponding to the actual weights. So it should work with a 64x64 matrix and 64x1 vector outputting a 64x1 vector as the maximum. It will need flexibility to handle sizes smaller than that as well.
  - According to ChatGPT, time-multiplexed will be slower than a tiled systolic array, but should have a smaller area and lower power requirements, as well as simpler logic. It is also well-suited for matrix-vector operations specifically, while a tiled systolic array would be a better choice if in the future, matrix-matrix operations were required.
- Create a testbench and perform functional verification for the above matrix-vector multiplier. Output will be also be streamed to prevent synthesis of a large number of wires for the output.
- Run OpenLane2 on the above matrix-vector multiplication module to determine timing and power requirements. From there, it may make sense to recalculate the target execution time given the change in SW / HW boundary and see if the matrix-vector multiplication alone results in an improvement. If it does not, determine where the bottleneck is, if it's the number of transfers due to time-multiplexing or some other factor.
  - May also consider whether it could make sense to incorporate the matrix-vector multiplication into a module that performs each gate calculation in total. For example, having a module or even chiplet for each of the below computations might make sense, if the input values could be shared:
  ```python
  i = vector_sigmoid(
      elementwise_add(
          matvec_mul(self.W_i, x),
          elementwise_add(matvec_mul(self.U_i, h_prev), self.b_i),
      )
  )
  f = vector_sigmoid(
      elementwise_add(
          matvec_mul(self.W_f, x),
          elementwise_add(matvec_mul(self.U_f, h_prev), self.b_f),
      )
  )
  c_tilde = vector_tanh(
      elementwise_add(
          matvec_mul(self.W_c, x),
          elementwise_add(matvec_mul(self.U_c, h_prev), self.b_c),
      )
  )
  o = vector_sigmoid(
      elementwise_add(
          matvec_mul(self.W_o, x),
          elementwise_add(matvec_mul(self.U_o, h_prev), self.b_o),
      )
  ```
