# ECE 510: Hardware for Artifical Intelligence and Machine Learning

This repository encapsulates work done related to ECE 510, with individual directories corresponding to weekly challenges. Each directory contains a README.md with additional details and links to other resources as needed.

An annotated bibliography covering all papers read for this class is below.

## Table of Contents

**Challenge 2: Paper Review for _Intrinsic and Designed Computation_ (2010)**  
Summary and analysis of Crutchfield, Ditto, & Sinha 2010 paper _Intrinsic and
Designed Computation_. ([link](./challenge-2/README.md))

**Challenge 5: Python Algorithm Analysis**  
Bytecode analysis and basic profiling for Python algorithms including quicksort, matrix multiplication, different equation solver, basic cryptography, and a convolutional neural network. ([link](./challenge-5/README.md))

**Challenges 6, 7, and 8: Simple Neuron and MLP Implementations**  
Includes implementation of a simple neuron with NAND and XOR functions (Challenge 6), visualization of the perceptron learning rule (Challenge 7), and implementation of a multi-layer perceptron (MLP) network trained to solve XOR (Challenge 8). ([link](./challenge-6/README.md))

**Challenges 9 and 12: Bootstrapping the Main Project and SW / HW Boundary**  
Includes training an LSTM Autoencoder model and creation of a pure-Python inference script (Challenge 9) and profiling and analysis of script to determine software / hardware boundary and target execution time for hardware acceleration (Challenge 12). ([link](./challenge-9/README.md))

**Challenges 10 and 11: Identification of Computational Bottlenecks and GPU Acceleration**  
Identification and discussion of computational bottlenecks in a FrozenLake Q learning algorithm (Challenge 10) and optimization of the algorithm of execution on a GPU (Challenge 11). ([link](./challenge-10/README.md))

**Challenge 15: Hardware Implementation of Main Project**  
Creation of SystemVerilog modules corresponding to initial hardware chosen for implementation based on the SW / HW boundary determined in Challenges 9 and 12. ([link](./challenge-15/README.md))

**Challenge 17: Sorting on a Systolic Array**  
Parallelized bubble sort implementation using a 1D systolic array. ([link](./challenge-17/README.md))

**Challenge 18: Going to the Transistor Level Using OpenLane2**  
Initial attempt to generate a physical transistor configuration from the HDL main project SystemVerilog code. ([link](./challenge-18/README.md))

**Challenge 22: Review and Discussion of _Neuromorphic computing at scale_ (2025)**  
(In progress) Discussion of Kudithipudi, Schuman, & Vineyard 2025 paper _Neuromorphic computing at scale_. ([link](./challenge-22/README.md))

## Week 7 Checkpoint

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

### Autoencoder Model Detail

The current autoencoder model is composed of 2 encoder and 2 decoder LSTM layers. Encoder 1 has a hidden size of 64, encoder 2 a hidden size of 32. Decoder 1 has a hidden size of 32, decoder 2 a hidden size of 64. There is a repeater layer in between the decoders and encoders, and a dense layer for a final linear transformation.

Six features were included from the NOAA dataset: air temperature, sea surface temperature, sea level pressure, wind speed, wave height, and wave period.

The inference algorithm is composed of 2 encoder and 2 decoder LSTM layers, and a final dense layer.

### Possible Changes in Next Iteration

#### Try a different SW / HW boundary at lower level

It may make sense to begin hardware acceleration at the matrix multiplication level, or perhaps at the level of computation for each gate within the LSTM Cell `step`. Each gate computation is composed of an activation function, multiple elementwise operations, and multiple matrix multiplication operations.

#### Consider simplifying model

Reducing the number of layers from 2 each for encoders and decoders to 1 each would significantly reduce the overall size of the weights. Within each cell, reducing the number of hidden neurons from 64 to 32 would reduce the weight for an individual gate.

Beyond that, reducing the number of features from 6 to 2 would reduce the overall sizing.

#### Reduce precision in hardware

The current model uses 32-bit precision. It could be worth attempting to use 16-bit precision in order to reduce the input and output wires required for the modules at the hardware level.

### Next Steps

- Recreate a simplified model with 1 encoder and 1 decoder, each with 32 hidden cells. Use 2 features instead of 6: air temperature and sea level pressure.
- Recreate inference algorithm as needed with new model. Need to determine the sizing of the input and output vectors. Benchmark new algorithm.
- Output weights and golden vectors.
- Start with creating a SV module for the matrix multiplication. Need to determine required input and output size for the matrix multiplier. Consider how to reduce precision, and impact of this change.
- Might try to get weights into SRAM with the above matrix multiplier, but may also just need to start by assuming they will be transferred per operation.
- Consider expanding hardware implementation to cover each gate computation - this would include the activation function applied to elementwise add on the result of matrix multiplication on the input kernel and the input with result of elementwise add of the matrix multiplication of the recurrent kernel and the previous hidden state with the bias (see `lstm_core.py` `step` method).

## Annotated Bibliography

**[1]** J. Wang, Z. Zhao, Z. Wang, and H. Jiang, “Designing Silicon Brains using LLM: Leveraging ChatGPT for Automated Description of a Spiking Neuron Array,” arXiv preprint arXiv:2402.10920, Feb. 2024. [Online]. Available: https://arxiv.org/abs/2402.10920

*This paper presents a process for guiding ChatGPT4 in production of a synthesizable hardware description of a programmable Spiking Neuron Array ASIC. Authors prompted ChatGPT to produce modules for a leaky integrate and fire (LIF) neuron, a network composed of LIF neurons, and a SPI implementation.*

*Through the process, code errors from ChatGPT were cataloged, which included both logical and syntax errors. Authors noted possible conflation on the part of ChatGPT between Verilog and SystemVerilog. After correcting the errors, the authors ultimately produced a design that was submitted for fabrication via TinyTapeout 5.*

**[2]** G. De Micheli and R. K. Gupta, “Hardware/software co-design,” Proceedings of the IEEE, vol. 85, no. 3, pp. 349–365, Mar. 1997, doi: 10.1109/5.558708.

*Authors describe the concurrent design of hardware and software as a means to achieve system-level objectives, termed hardware/software co-design. They describe application of co-design across multiple domains, including embedded systems, Instruction Set Architectures (ISAs), and reconfigurable systems such as FPGAs. Distinctions between the different domains, such as degree of programmability and levels of programming, and their unique challenges in co-design are discussed. Authors also provide recommendations for determining the partition between software and hardware, and briefly describe task scheduling considerations.*

*A few insights of note include the economic considerations when determining the partitioning between hardware and software at different points in a product's lifecycle. For example, authors note that an early-stage product may contain a higher proportion of functionality implemented in software, in order to allow for shorter time to market and greater flexibility; as the product matures and functionality becomes set, shifting functionality to hardware can improve product performance.*

*Authors also describe ISAs as the boundary between hardware and software, stating that it provides "a programming model of the hardware." The goal of an ISA is to fully utilize the underlying hardware, and as such, compiler design and development should take place early in ISA development.*

**[3]** C. Guo et al., “A Survey: Collaborative Hardware and Software Design in the Era of Large Language Models,” IEEE Circuits and Systems Magazine, vol. 25, no. 1, pp. 35–57, 2025, doi: 10.1109/MCAS.2024.3476008.

*This article provides a wide-ranging survey of hardware and software strategies for meeting the current challenges in Large Language Model (LLM) development, focusing particularly on the data transfer and consumption of most LLMs. Authors distinguish between the more intensive demands of training LLMs versus the less-intensive but still significant performance challenges in inference.*

*Accelerators for training mainly focus on memory-centric optimizations, including using mixed precision data types (e.g. combinations of 16- and 32-bit floating point), systolic array architecture in a Tensor Processing Unit to accelerate matrix multiplication, and near-storage ASIC accelerators. Inference performance improvements also include memory optimizations, such as using a paged model for key-value cache data, as well as algorithmic efficiencies that reduce the required data with minimal impact on model correctness. Authors also discuss both hardware and algorithmic acceleration for inference, including compute-in-memory (CIM) architectures and various compression techniques.*

**[4]** J. P. Crutchfield, W. L. Ditto, and S. Sinha, “Introduction to Focus Issue: Intrinsic and Designed Computation: Information Processing in Dynamical Systems—Beyond the Digital Hegemony,” Chaos, vol. 20, no. 3, p. 037101, Sep. 2010, doi: 10.1063/1.3492712.

*This article provides a high-level overview of intrinsic computation, an emergent characteristic of dynamical systems related to their information storage and processing which produces resultant behavior and organization. This form of computation is in contrast with designed computation, which has some utility or useful behavior for its designers. Authors discuss historical work related to chaotic dynamical systems, organized complexity, cybernetics, and information theory, among other topics.*

*See [challenge-2/README.md](./challenge-2/README.md) for additional review.*

**[5]** Ş. M. Kaya, B. İşler, A. M. Abu-Mahfouz, J. Rasheed, and A. AlShammari, “An Intelligent Anomaly Detection Approach for Accurate and Reliable Weather Forecasting at IoT Edges: A Case Study,” Sensors, vol. 23, no. 5, p. 2426, 2023, doi: 10.3390/s23052426.

*The authors present a brief overview of existing work related to weather forecasting and analysis using remote sensor networks. They present their experimental work testing five different machine learning (ML) algorithms in anomaly detection within weather data, with the goal of filtering out invalid data at a preprocessing stage. Algorithms tested included support vector classification (SVC), Adaboost, logistic regression (LR), naive Bayes (NB), and random forest (RF). They found that most algorithms had a high degree of accuracy but varied in latency.*

*All algorithms except NB displayed perfect accuracy on the defined anomaly detection task, suggesting that the task itself was relatively trivial. However, the Edge compute aspects of the study do provide an interesting model in consideration of more complex ML workloads in embedded systems.*

**[6]** U. Rawal and S. Patel, “Anomaly Detection in Meteorological Data Using Machine Learning Techniques,” in Proc. 2025 IEEE Int. Students’ Conf. Electr., Electron. Comput. Sci. (SCEECS), Bhopal, India, 2025, doi: 10.1109/SCEECS64059.2025.10940145.

*Authors provide a thorough literature review covering work related to identifying strengths of particular algorithms in varying contexts in the task of anomaly detection in weather data. They then perform anomaly detection on a 3-feature dataset using five different algorithms: Density Based Spatial Clustering of Applications with Noise (DBSCAN), Isolation Forest (IF), Local Outlier Factor (LOF), Elliptic Envelope, and One Class Support Vector Machine (One-Class SVM). Results are compared for each algorithm.*

*The authors found different strengths for each algorithm. In particular, DBSCAN performed well with dense clusters of data. IF did well with high dimensionality dataset, but hyper parameter tuning was necessary. LOF, One-Class SVM, and Elliptic Envelope each had noted strengths and weaknesses, and particular tuning or optimization requirements.*

*Also of note, the authors cite another paper which discusses using hybrid ML models in order to combine the best features of each.*

**[7]** Y. R. Jeong, K. Cho, Y. Jeong, S. B. Kwon, and S. E. Lee, “A Real-Time Reconfigurable AI Processor Based on FPGA,” in Proc. 2023 IEEE Int. Conf. Consum. Electron. (ICCE), Las Vegas, NV, USA, 2023, pp. 1–2, doi: 10.1109/ICCE56470.2023.10043575.

*The authors begin by presenting motivation for reconfigurable hardware in an AI computation context, citing the differing hardware and software demands depending on the target application. They note that while software has an inherent flexibility, providing similar flexibility in hardware has unique challenges. They then describe a reconfigurable processor architecture they created using an FPGA as the primary compute module with a supporting MCU to facilitate reconfiguration. Configuration data is stored in configuration flash memory (CFM) and loaded into configuration RAM (CRAM) during synthesis. The MCU initiates reconfiguration, which may be performed remotely. Authors compare AI performance across two different target applications, showing differing accuracy depending on the configured AI processor, supporting the initial claim that hardware reconfiguration may be necessary to achieve peak performance.*

**[8]** Y. Kwon et al., “Chiplet Heterogeneous-Integration AI Processor,” in Proc. 2023 Int. Conf. Electron., Inf., Commun. (ICEIC), Singapore, 2023, pp. 1–2, doi: 10.1109/ICEIC57457.2023.10049867.

*Authors describe a generalized chiplet architecture with specific application for AI workloads, which require significant data processing and transfer between components. They contrast the chiplet design containing multiple dies on a single chip with IP-based design, stating that the chiplet approach may provide higher performance with lower cost. They provide an overview of the components required in a chiplet functioning as an AI processor, which include multiple Neural Processing Units (NPUs), High Bandwidth Memory (HBM), an interposer to provide connections between components, and links between NPUs.*

*Authors provide a performance equation showing the factors impacting max performance, which include the memory transaction efficiency, the data transfer efficiency, and the intra-die architectural efficiency as scaling factors for the peak performance. They go on to discuss specific physical aspects in chiplet design which determine these efficiency factors. In one example, they explain how the reduced-pitch bumps on the HBM dies allow for greater density of connections, increasing bandwidth. They also discuss impacts of thermal expansion, where repeated cycles of heating and cooling produce bonding failures that require mitigation. They conclude by stating that the chiplet is a viable approach for performant compute in highly data-intensive AI workloads, and reiterate the primary design considerations in achieving high performance.*


**[9]** D. Kudithipudi, C. Schuman, C. M. Vineyard, et al., “Neuromorphic computing at scale,” *Nature*, vol. 637, pp. 801–812, 2025, doi: 10.1038/s41586-024-08253-8.

*The authors in this article describe unique features of neuromorphic computing systems as compared to traditional von Neumann architectures and more recently developed deep learning accelerator models. They present both the distinct advantages and challenges of neuromorphic models, in particular as they relate to large-scale deployments. They provide a broad overview of the neuromorphic computing ecosystem, and note distinct areas of focus for future development.*

*Through a comparison to mainstream AI toolchains, they identify specific components that are currently lacking in neuromorphic toolchains. More generally, they list multiple areas in which standardization could play a role in increased adoption, which may be a key factor in developing the necessary workflows and platforms for large-scale implementations. A specific area for development the authors highlight is in high-level software tooling, which would abstract the lower-level computation components, analogous to tols like PyTorch and TensorFlow for AI.*

*The authors conclude by listing three categories of questions to prompt further development, organized by the timeline in which they should be addressed. These questions help to inform a roadmap for bringing neuromorphic computation to greater maturity and broader usage.*
