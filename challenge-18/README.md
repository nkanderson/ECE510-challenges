# Going to the Transistor Level Using OpenLane2

**Niklas Anderson**  
**Spring 2025**

## Overview

Initial attempt to generate a physical transistor configuration from the HDL main project SystemVerilog code.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md).

## Challenges

- Very long runtimes for openlane
- Failures at certain stages
- Manual creation of lef file, difficulty continuing run from failed stage
- Ran through synthesis with a smaller module from Challenge 17 to attempt to achieve a successful runthrough. After that was successful and requesting analysis from ChatGPT, determined there were probably significant barriers to getting the current iteration to synthesis properly.

## Results

Did not complete synthesis on the first iteration of SV code, but similar to in Challenge 15, there were lessons learned along the way. Was able to complete synthesis on a smaller module, proving the toolchain was set up correctly, and making it clearer that the current iteration needed modification.
