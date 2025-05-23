# RTL Implementation of Main Project

**Niklas Anderson**  
**Spring 2025**

## Overview

Creation of SystemVerilog modules corresponding to initial hardware chosen for implementation based on the SW / HW boundary determined in Challenges 9 and 12.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md).

## Challenges

- Extracting weights and golden vectors
- Weight loader
- Particular SV rules that ChatGPT has a hard time following, e.g. no blocking assignment in `always_ff`

## Results

Did not complete functional verification, but got a much better understanding of what might be necessary to get the model weights in SV code. Worked on this process in parallel with Challenge 18, so after determining that this implementation would not likely succeed, decided to start with a new HW / SW boundary in the second iteration, knowing the first iteration SV code could be used for reference.
