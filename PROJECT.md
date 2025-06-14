# LSTM Autoencoder Hardware Acceleration Project Report

**Niklas Anderson**  
**Spring 2025**  

**NOTE: This document is in-progress. Main sections have been outlined below, with basic lists of intended contents.**

## Overview

- Installation instructions
- Quickstart
- Diagrams

## Iteration One

- Link to related challenges (9, 12, 15, 18)
- Discuss failure of functional verification, but benefit of moving on to iteration 2 after working on synthesis in parallel (and more quickly discovering design issues preventing synthesis)

## Version 0.2.0

- Implementation and changes from iteration one (reference challenge 23 checkpoint document, and README from lstm_autoencoder-v0.2.0 directory)
- Testing - passed functional verification, but discuss improvements to be made here
  - Images from simulation

## Design Decisions

- Time-multiplexed over systolic array - optimizing for reduced size and power consumption over execution time
- Work to tailor the hardware for the specific weights and input vector values - created scripts to find the max and min values and used fixed point formats based on these values
- Weight-stationary in SRAM - very important for reducing communication cost, but a significant barrier in synthesis. Will be further discussed later.

## Results & Discussion

- Based on estimates (reference work from challenge 12), the proof-of-concept v0.2.0 would meet the threshhold for providing a speedup
  - Reference simulation results showing number of clock cycles as well
- Analysis of metrics.json indicates this design successfully prioritized relatively low power consumption and a small footprint, in alignment with initial goals for deployment in a remote environment, over execution speed
  - Ideal comparison would be to the CPU execution, might try to look into making that determination
  - As mentioned in the v0.2.0 README, the analysis was not performed on a synthesized module using SRAM macros. With SRAM macros successfully integrated into the design, the power, die area, and number of transistors will inevitably increase. This may make it more important to adjust the software/hardware boundary again to make better use of the fact that the weights are stored in SRAM by including more functionality in hardware.
- Challenges, especially related to limited domain expertise in initial model creation
  - Naive LLM-driven implementation allowed for relatively quick iteration, but at the cost of making mistakes that likely could have been avoided with better conceptual understanding of the model beforehand. A deeper understanding was ultimately gained, but note the tradeoffs.

## Future Work

- Find or build an SRAM macro that works with openlane2, or find another synthesis toolchain that has usable SRAM macros
- Reference list in lstm_autoencoder-v0.2.0 README for details on SRAM memory layout and FSM
- Adjusting the software/hardware boundary to leverage possible parallelism at the gate level (i.e. 4 gates f, i, o, c may compute values in parallel based on `step` function)
- Multiple MAC4 instances
- Make MAC4 into sequential pipelined logic to avoid long critical path in combinational version
  - Is this module the bottleneck? How to determine?
- cocotb testing with simulated SPI bus to verify timing estimates
