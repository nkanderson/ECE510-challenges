# ECE 510: Hardware for Artifical Intelligence and Machine Learning

This repository encapsulates work done related to ECE 510, with individual directories corresponding to weekly challenges. Each directory contains a README.md with additional details and links to other resources as needed.

An annotated bibliography covering all papers read for this class is below.

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
