# Going to the Transistor Level Using OpenLane2

**Niklas Anderson**  
**Spring 2025**

## Overview

In this challenge, an initial attempt was made to generate a physical transistor configuration from the HDL main project SystemVerilog code. Synthesis was not successful, but familiarity was gained with the synthesis toolchain, and potential issues in the design were identified for alteration in iteration two.

Full LLM transcripts found in [LLM_TRANSCRIPT.md](./docs/LLM_TRANSCRIPT.md). Includes three threads with ChatGPT, two of which focused on usage and troubleshooting of OpenLane2, and the third on identifying issues with the design. The first is an attempt to troubleshoot a failed LEF generation, and the second is an attempt to explore the GUI with a smaller model that completed synthesis. The third asks for optimization suggestions for the existing design, and results were used to help inform iteration two of the design.

## Synthesis with OpenLane2

To begin, the [SystemVerilog code from Challenge 15](../challenge-15/) was converted to Verilog using the tool [sv2v](https://github.com/zachjs/sv2v), which is explicitly for the purpose of generating synthesizable code. OpenLane2 was installed as described in their [documentation](https://openlane2.readthedocs.io/en/latest/getting_started/newcomers/index.html#installation), and the provided basic example was synthesized. From there, using the example `config.json`, a project-specific `config.json` was created and the command `openlane config.json` was run from within the `nix-shell` `openlane` environment.

Synthesis failed at a number of stages. The first of which required an increase in the `DIE_AREA` in `config.json`. The `openlane` command took a significant amount of time to run through steps such as stage 6, where `yosys` synthesis was performed. Eventually there were consistent failures at stage 59, and troubleshooting was initiated.

## Challenges

The very long runtimes for OpenLane2 made it difficult to test different synthesis configuration, as the feedback loop was a significant impediment to timely iteration. Working with ChatGPT, attempts were made to determine how to restart the `openlane` command from a failed stage, rather than starting from the beginning. However, ChatGPT did not seem to be familiar with version 2 of OpenLane, or with the `nix-shell` installation recommended in OpenLane2 docs. Despite working around the failure at a certain stage by manual generation of an LEF file, it was not possible to re-run starting with the stage that failed due to the missing file.

## Results

Better understanding of the `openlane` toolchain was gained through this challenge, though synthesis was not successful for the first iteration of SystemVerilog code for the project. Synthesis was completed for a smaller module as a means of ensuring the toolchain was working, and to better understand what success might look like. This also served as an indication that the existing design needed alterations in order to be synthesized.

The final thread with ChatGPT was focused on identifying specific changes to make in the next iteration of the design. The factors discussed were taken into consideration in the checkpoint document for [Challenge 23](../challenge-23/README.md), as well as the [second iteration of the SystemVerilog code](../lstm_autoencoder-v0.2.0/sv/).
