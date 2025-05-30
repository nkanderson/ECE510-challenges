# Changelog

Tracking changes made from the initial iteration to this second iteration (version 0.2.0).

**May 16, 2025**

Fixed an issue with the weight loading. The weights from the saved model `lstm_weights.npz` required transposition in order to be accurate. This was discovered when it was determined that the `tanh` sigmoid activation was being passed empty vectors of values.

Inference steps were re-run and produced results indicating greater accuracy for model reconstruction using both the training and novel data.

**May 30, 2025**

Fixed the `matvec_mul` SystemVerilog module to properly load a vector provided by a client and matrix data of varying sizes from a `matrix_loader` module, then return calculated values for resulting vector one row at a time. Includes basic verification that may be extended in the future. Uses hardcoded matrix data values from the `matrix_loader`, and the next step will be to replace this with actual weights.

This version is functional, but lacks efficiency. Future iterations should incorporate pipelining, so that calculations are performed simultaneously with loading of matrix data.
