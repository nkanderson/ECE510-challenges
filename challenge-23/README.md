# Week 7 Checkpoint

This checkpoint serves to document current status of the project as it relates to the definition of project success described in [w7_codefest.pdf](./w7_codefest.pdf), as well as outline next steps to achieve improvement in the next iteration. At a high level, all stages of the project have been attempted, though with varying success. Specifically, challenges have come up in functional verification of the model in SV, and the transistor-level design using OpenLane2 has not successfully completed. However, a better understanding of the requirements for each stage has been gained. The first iteration may be described as a relatively naive LLM-driven attempt, but it has been useful in gaining a better understanding of the project at all levels.

Current status:
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

## Autoencoder Model Detail

The current autoencoder model is composed of 2 encoder and 2 decoder LSTM layers. Encoder 1 has a hidden size of 64, encoder 2 a hidden size of 32. Decoder 1 has a hidden size of 32, decoder 2 a hidden size of 64. There is a repeater layer in between the decoders and encoders, and a dense layer for a final linear transformation.

Six features were included from the NOAA dataset: air temperature, sea surface temperature, sea level pressure, wind speed, wave height, and wave period.

The inference algorithm is composed of 2 encoder and 2 decoder LSTM layers, and a final dense layer.

## Possible Changes in Next Iteration

### Try a different SW / HW boundary at lower level

It may make sense to begin hardware acceleration at the matrix multiplication level, or perhaps at the level of computation for each gate within the LSTM Cell `step`. Each gate computation is composed of an activation function, multiple elementwise operations, and multiple matrix multiplication operations.

### Consider simplifying model

Reducing the number of layers from 2 each for encoders and decoders to 1 each would significantly reduce the overall size of the weights. Within each cell, reducing the number of hidden neurons from 64 to 32 would reduce the weight for an individual gate.

Beyond that, reducing the number of features from 6 to 2 would reduce the overall sizing.

### Reduce precision in hardware

The current model uses 32-bit precision. It could be worth attempting to use 16-bit precision in order to reduce the input and output wires required for the modules at the hardware level.

## Next Steps

- Recreate a simplified model with 1 encoder and 1 decoder, each with 32 hidden cells. Use 2 features instead of 6: air temperature and sea level pressure.
- Recreate inference algorithm as needed with new model. Need to determine the sizing of the input and output vectors. Benchmark new algorithm.
- Output weights and golden vectors for the new model.
- Estimate the required input and output sizes necessary in hardware for matrix multiplication and decide on an implementation that will be feasible to synthesize. Consider creating a small version and running through OpenLane before expanding.
- Consider reducing precision for hardware, starting with the matrix multiplication, and impact of this change.
- Might try to get weights into SRAM with the above matrix multiplier, but may also just need to start by assuming they will be transferred per operation.
- Consider expanding hardware implementation to cover each gate computation - this would include the activation function applied to elementwise add on the result of matrix multiplication on the input kernel and the input with result of elementwise add of the matrix multiplication of the recurrent kernel and the previous hidden state with the bias (see `lstm_core.py` `step` method).
