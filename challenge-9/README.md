# Bootstrapping the Main Project

## Heilmeier Questions

Questions from [the Heilmeier Catechism](https://www.darpa.mil/about/heilmeier-catechism):

1. What are you trying to do? Articulate your objectives using absolutely no jargon. 

I want to create a hardware accelerator for anomaly detection in weather data. Specifically, with this hardware accelerator, it should be possible to run AI workloads on sensor data in remote locations. Results could then be transmitted within the region to allow for timely response to changing conditions or critical weather events.

1. How is it done today, and what are the limits of current practice? 

Typically, this is currently done at a centralized compute location, after data has been transferred and aggregated. One of the main limitations is the significant data transfer involved. The data transfer and subsequent centralized analysis leads to latency in communicating the dangers of critical weather events.

1. What is new in your approach and why do you think it will be successful? 

In my approach, it should be possible to deploy remote inference stations that would be near to sensors collecting weather data. These stations could collect data from a handful of nearby sensors and perform inference after being trained on region-specific data. Rather than transmitting all sensor data back to a centralized location, these stations could transmit the model output locally, indicating whether conditions for a critical weather event (for example, a wildfire) have been detected.

1. Who cares? If you are successful, what difference will it make? 

If successful, this could open the door to having better early-warning systems for critical weather events. This could reduce the negative impacts of such events, allowing for greater planning and protection of people and property, and evacuation if necessary. It could help in resource management for such events by providing accurate information on which regions are most immediately at risk.

1. What are the risks? 

One of the risks is that the accelerator may not meet the efficiency requirements for low-power environments. Some environmental sensor stations are extremely power-limited (for example, a [Pico Balloon(https://en.wikipedia.org/wiki/High-altitude_balloon#Amateur_radio_high-altitude_ballooning)]), and it may be difficult to reduce power consumption enough to meet requirements, even using a hardware accelerator. However, there are other types of weather stations that may accommodate slightly greater power needs.

Another risk is in development of accurate prediction models for a given region, and the use of available sensor network data. To train such models, regional data needs to be available, along with appropriate interpretation from subject-matter experts. Though the algorithm I'm looking at using leverages unsupervised training, verification of results should be possible prior to deployment. In addition, these models will ideally be able to use currently-available sensor data, rather than requiring new or more expensive types of sensor data.

1. How much will it cost? 

Initial proof-of-concept work will be fairly limited in cost, but may require significant time commitment. Later in development, it may be possible to reduce costs by piggybacking onto other weather or environmental sensor projects by providing an extension to the existing functionality, instead of attempting to create a new sensor network from scratch dedicated to this project.

If initial development shows promise, wide-scale deployment would incur costs related to the hardware manufacturing for the hardware accelerator. In addition, there would be costs related to software development and integration of the new hardware into existing or new sensor network projects.

1. How long will it take? 

Exploratory work has already been done in identifying the bottlenecks for anomaly detection in weather data using the Long Short-Term Memory (LSTM) autoencoder algorithm written in Python. Estimation of communication costs and development of a target execution time are currently in progress. From here, creating basic hardware modules for the necessary components in SystemVerilog may be done over the next week or two. After that, 4-6 weeks should be reserved for iteration and refinement of the hardware, with the goal of creating a synthesizable design by week 8.

1. What are the mid-term and final “exams” to check for success? 

A mid-term checkpoint should contain a basic hardware simulation using SystemVerilog. At that point, algorithm bottlenecks will have been identified, and communication costs associated with the use of the hardware accelerator as a co-processor will be estimated. This will provide a target execution time for the hardware, based on the maximum execution time it may take while providing an improvement over the software-only approach.

A final exam would ideally result in a synthesizable chiplet providing hardware acceleration for the specific parts of the chosen algorithm designated as a bottleneck. Estimated execution time, including communication costs, should be less than the execution time of the software alone.

## LSTM Model

### Training

Got the following output from running the `train_lstm_autoencoder` script on the preprocessed data:

```sh
Training on 476 sequences, shape: (24, 6)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 24, 64)              │          18,176 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 32)                  │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ repeat_vector (RepeatVector)         │ (None, 24, 32)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (LSTM)                        │ (None, 24, 32)              │           8,320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_3 (LSTM)                        │ (None, 24, 64)              │          24,832 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ time_distributed (TimeDistributed)   │ (None, 24, 6)               │             390 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 64,134 (250.52 KB)
 Trainable params: 64,134 (250.52 KB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 3s 41ms/step - loss: 0.1224 - val_loss: 0.0351
Epoch 2/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0444 - val_loss: 0.0305
Epoch 3/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0337 - val_loss: 0.0251
Epoch 4/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0280 - val_loss: 0.0233
Epoch 5/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0265 - val_loss: 0.0219
Epoch 6/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0246 - val_loss: 0.0218
Epoch 7/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.0239 - val_loss: 0.0202
Epoch 8/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0231 - val_loss: 0.0203
Epoch 9/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0225 - val_loss: 0.0198
Epoch 10/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0221 - val_loss: 0.0196
Epoch 11/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0216 - val_loss: 0.0197
Epoch 12/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0218 - val_loss: 0.0192
Epoch 13/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0214 - val_loss: 0.0191
Epoch 14/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.0214 - val_loss: 0.0193
Epoch 15/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0212 - val_loss: 0.0189
Epoch 16/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0212 - val_loss: 0.0188
Epoch 17/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0211 - val_loss: 0.0190
Epoch 18/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0212 - val_loss: 0.0187
Epoch 19/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0208 - val_loss: 0.0186
Epoch 20/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0207 - val_loss: 0.0185
Epoch 21/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0209 - val_loss: 0.0185
Epoch 22/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0204 - val_loss: 0.0184
Epoch 23/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0205 - val_loss: 0.0185
Epoch 24/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0207 - val_loss: 0.0183
Epoch 25/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0205 - val_loss: 0.0185
Epoch 26/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0207 - val_loss: 0.0186
Epoch 27/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0205 - val_loss: 0.0183
Epoch 28/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0204 - val_loss: 0.0185
Epoch 29/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.0205 - val_loss: 0.0182
Epoch 30/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0204 - val_loss: 0.0182
Epoch 31/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0205 - val_loss: 0.0182
Epoch 32/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0202 - val_loss: 0.0182
Epoch 33/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0201 - val_loss: 0.0184
Epoch 34/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0200 - val_loss: 0.0181
Epoch 35/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0199 - val_loss: 0.0183
Epoch 36/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0200 - val_loss: 0.0182
Epoch 37/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0200 - val_loss: 0.0181
Epoch 38/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0201 - val_loss: 0.0181
Epoch 39/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0206 - val_loss: 0.0183
Epoch 40/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0204 - val_loss: 0.0182
Epoch 41/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0201 - val_loss: 0.0180
Epoch 42/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0201 - val_loss: 0.0181
Epoch 43/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0198 - val_loss: 0.0181
Epoch 44/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0200 - val_loss: 0.0180
Epoch 45/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0200 - val_loss: 0.0178
Epoch 46/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0199 - val_loss: 0.0178
Epoch 47/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0199 - val_loss: 0.0181
Epoch 48/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0198 - val_loss: 0.0179
Epoch 49/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0199 - val_loss: 0.0177
Epoch 50/50
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0196 - val_loss: 0.0177
Model saved to: saved_model/lstm_autoencoder.keras
Weights saved to: saved_model/lstm_weights.npz
Scaler saved to: saved_model/scaler.gz
```

### Commands

For inference, there are multiple modes. Running inference on the original data is the default:
```sh
python lstm_inference.py
```

Running inference on new data which has been normalized can be accomplished with the following:
```sh
python lstm_inference.py --npz new_data_sequences.npz
```

Plotting data can be done with the following:
```sh
python lstm_inference.py --plot
```

CSV output mode is available as well, and the following command will output the MSE values to a CSV:
```sh
python lstm_inference.py --csv
```

Output a CSV with the window index and MSE for anomalies, as determined by the 95% threshhold:
```sh
python lstm_inference.py --anomalies
```

To trace anomalies back to raw data:
```sh
python extract_anomalies.py
```

### Profiling and Analysis

Generate a prof file to use with snakeviz:
```sh
python -m cProfile -o profile.lstm.prof lstm_inference.py --npz new_data_sequences.npz
snakeviz profile.lstm.prof
```

Results from using `kernprof` (with the actual MSE inference output removed):
```sh
(venv) ➜  challenge-9 git:(main) ✗ kernprof -l -v lstm_inference.py --npz new_data_sequences.npz

Wrote profile results to lstm_inference.py.lprof
Timer unit: 1e-06 s

Total time: 2.03411 s
File: lstm_inference.py
Function: matvec_mul at line 41

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    41                                           @profile
    42                                           def matvec_mul(matrix, vec):
    43   2771496    2034109.0      0.7    100.0      return [sum(m * v for m, v in zip(row, vec)) for row in matrix]

Total time: 2.97621 s
File: lstm_inference.py
Function: step at line 64

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    64                                               @profile
    65                                               def step(self, x, h_prev, c_prev):
    66     52416     212154.0      4.0      7.1          i = vector_sigmoid(
    67     52416      78333.0      1.5      2.6              elementwise_add(
    68     26208     567409.0     21.7     19.1                  matvec_mul(self.W_i, x),
    69     26208    1509230.0     57.6     50.7                  elementwise_add(matvec_mul(self.U_i, h_prev), self.b_i),
    70                                                       )
    71                                                   )
    72     52416      15441.0      0.3      0.5          f = vector_sigmoid(
    73     52416      18251.0      0.3      0.6              elementwise_add(
    74     26208     197643.0      7.5      6.6                  matvec_mul(self.W_f, x),
    75     26208      53441.0      2.0      1.8                  elementwise_add(matvec_mul(self.U_f, h_prev), self.b_f),
    76                                                       )
    77                                                   )
    78     52416      14827.0      0.3      0.5          c_tilde = vector_tanh(
    79     52416      17703.0      0.3      0.6              elementwise_add(
    80     26208      41937.0      1.6      1.4                  matvec_mul(self.W_c, x),
    81     26208      52655.0      2.0      1.8                  elementwise_add(matvec_mul(self.U_c, h_prev), self.b_c),
    82                                                       )
    83                                                   )
    84     52416      14810.0      0.3      0.5          o = vector_sigmoid(
    85     52416      18029.0      0.3      0.6              elementwise_add(
    86     26208      42215.0      1.6      1.4                  matvec_mul(self.W_o, x),
    87     26208      52662.0      2.0      1.8                  elementwise_add(matvec_mul(self.U_o, h_prev), self.b_o),
    88                                                       )
    89                                                   )
    90     26208      35936.0      1.4      1.2          c = elementwise_add(elementwise_mul(f, c_prev), elementwise_mul(i, c_tilde))
    91     26208      20916.0      0.8      0.7          h = elementwise_mul(o, vector_tanh(c))
    92     26208      12620.0      0.5      0.4          return h, c

Total time: 0.318005 s
File: lstm_inference.py
Function: forward at line 101

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   101                                               @profile
   102                                               def forward(self, x):
   103      6552     318005.0     48.5    100.0          return elementwise_add(matvec_mul(self.W, x), self.b)

Total time: 3.46085 s
File: lstm_inference.py
Function: reconstruct at line 130

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   130                                               @profile
   131                                               def reconstruct(self, sequence):
   132       273         82.0      0.3      0.0          h1 = [0.0] * self.hidden1
   133       273         67.0      0.2      0.0          c1 = [0.0] * self.hidden1
   134       273         52.0      0.2      0.0          h2 = [0.0] * self.hidden2
   135       273         44.0      0.2      0.0          c2 = [0.0] * self.hidden2
   136                                           
   137      6825       1052.0      0.2      0.0          for x in sequence:
   138      6552     791068.0    120.7     22.9              h1, c1 = self.lstm1.step(x, h1, c1)
   139      6552     783411.0    119.6     22.6              h2, c2 = self.lstm2.step(h1, h2, c2)
   140                                           
   141      6825       1258.0      0.2      0.0          repeated = [h2 for _ in range(len(sequence))]
   142       273         72.0      0.3      0.0          h3 = [0.0] * self.hidden3
   143       273         52.0      0.2      0.0          c3 = [0.0] * self.hidden3
   144       273         57.0      0.2      0.0          h4 = [0.0] * self.hidden4
   145       273         52.0      0.2      0.0          c4 = [0.0] * self.hidden4
   146                                           
   147       273         30.0      0.1      0.0          output_seq = []
   148      6825       1199.0      0.2      0.0          for x in repeated:
   149      6552     627067.0     95.7     18.1              h3, c3 = self.lstm3.step(x, h3, c3)
   150      6552     927005.0    141.5     26.8              h4, c4 = self.lstm4.step(h3, h4, c4)
   151      6552     327349.0     50.0      9.5              y = self.dense.forward(h4)
   152      6552        752.0      0.1      0.0              output_seq.append(y)
   153                                           
   154       273        182.0      0.7      0.0          return output_seq
```

## Hardware Acceleration

### Hardware Execution Time

Estimating communication cost:
- Get baseline for a protocol like PCIe or similar
- Get estimate of amount of data that would be transferred to hardware
- The above should provide a basic estimate of communication cost
- From there, the existing software-only computational bottleneck minus the communication cost (round-trip) should represent a target execution time that the hardware should be faster than

For the `step` function, we can save on communication cost by saving the weights in SRAM. This should be estimated and considered as part of the whole hardware cost.

A few specific points from discussion with ChatGPT:

Chiplet optimizations that may improve data transfer efficiency:
- Holds weights (W, U, b) locally in SRAM → no transfer needed
- Chains layers internally (LSTM1 → LSTM2) → you save intermediate transfers
- Implements on-chip h/c buffering → only stream x in and h out

Best-Case Streamed Input:
- Just send x per timestep, receive h
- ~24 × (input_size + hidden_size) × 4 = much smaller
