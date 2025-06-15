# Simulation with cocotb

**Niklas Anderson**  
**Spring 2025**

## Overview

**This challenge is still in progress.**

Goal is to code an SPI interface between software and hardware, and perform a high-level co-simulation for the purposes of benchmarking performance.

A basic test using cocotb has been created for the smallest functional unit in the iteration two design, the `mac4` module. Next steps are to incorporate a SPI module (may consider using the implementation at [tom-urkin/SPI](https://github.com/tom-urkin/SPI)), attempt to test the `mac4` on the SPI bus, then build up to test the `matvec_mult` module with the [`lstm_inference.py`](../challenge-09/lstm_inference.py) script. A final working test should be placed within the [lstm_autoencoder-v0.2.0](../lstm_autoencoder-v0.2.0/) directory.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md).
