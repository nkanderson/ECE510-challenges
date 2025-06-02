# Changelog

Tracking changes made from the initial iteration to this second iteration (version 0.2.0).

**May 16, 2025**

Fixed an issue with the weight loading. The weights from the saved model `lstm_weights.npz` required transposition in order to be accurate. This was discovered when it was determined that the `tanh` sigmoid activation was being passed empty vectors of values.

Inference steps were re-run and produced results indicating greater accuracy for model reconstruction using both the training and novel data.

**May 30, 2025**

Fixed the `matvec_mul` SystemVerilog module to properly load a vector provided by a client and matrix data of varying sizes from a `matrix_loader` module, then return calculated values for resulting vector one row at a time. Includes basic verification that may be extended in the future. Uses hardcoded matrix data values from the `matrix_loader`, and the next step will be to replace this with actual weights.

This version is functional, but lacks efficiency. Future iterations should incorporate pipelining, so that calculations are performed simultaneously with loading of matrix data.


**May 31, 2025**

Updatted `matrix_loader` module to use [`sky130_sram_2kbyte_1rw1r_32x512_8`](https://github.com/VLSIDA/sky130_sram_macros/blob/main/sky130_sram_2kbyte_1rw1r_32x512_8/sky130_sram_2kbyte_1rw1r_32x512_8.v) SRAM macros. The macros are instantiated within `ifdef USE_SRAM_MACRO` to allow separate implementation for synthesis versus simulation.

The `matvec_mult` and `matrix_loader` were also updated to have flattened input and output arrays, rather than 2D arrays, to support efficient synthesis.
