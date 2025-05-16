# Changelog

Tracking changes made from the initial iteration to this second iteration (version 0.2.0).

**May 16, 2025**

Fixed an issue with the weight loading. The weights from the saved model `lstm_weights.npz` required transposition in order to be accurate. This was discovered when it was determined that the `tanh` sigmoid activation was being passed empty vectors of values.

Inference steps were re-run and produced results indicating greater accuracy for model reconstruction using both the training and novel data.
