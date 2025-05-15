# Paper Review: *Neuromorphic computing at scale*

**Niklas Anderson**  
**May 2025**

1. The authors discuss several key features necessary for neuromorphic systems at scale
(distributed hierarchy, sparsity, neuronal scalability, etc.). Which of these features do you
believe presents the most significant research challenge, and why? How might
overcoming this challenge transform the field?

Challenges in implementing distributed hierarchy include modularization of different components, as well as establishing a standardized communication protocol between components. It would likely require understanding the minimal data representation necessary for each component to produce and pass to other components. In other words, it would require establishing APIs between all the components. As noted, working towards this goal would increase explainability, which is a challenge in current AI models. Modularization has typically supported greater adoption, as it allows for specialization and abstraction.

Plasticity or dynamic reconfigurability is described by the authors as being in an early stage of maturity. They define requirements for this feature as being rooted in patterns seen in human cognition, including networks "that interact in complex and transient communication patterns"[1]. The huge design space for this particular feature seems to add to the challenges in its research. To even begin research, it is necessary to choose a particular focus, resulting in the exclusion of many other possible avenues. For example, at what level of plasticity would it make sense to begin with - the individual neurons, or the system as a whole? The authors note it may be either. If one were to choose the individual neurons, there is then the challenge of analyzing the massive number of possible combinations of neuron states as they interact with one another. It seems like the secondary challenge is that of analyzing combinatorial states, after the primary challenge of identifying what variable conditions to allow.

The power of neuronal plasticity seems apparent, if the challenges could be overcome. For example, it seems like having reconfigurable neurons could support other features such as heterogeneous functionality and hierarchical systems, though the ability to modify neuronal activity depending on the context. If reconfigurable neurons were generally available and had a straightforward API, it's easy to imagine much broader adoption, as deployment and usage would have greater flexibility.

2. The article compares neuromorphic computing's development to the evolution of deep
learning, suggesting it awaits its own "AlexNet moment." What specific technological or
algorithmic breakthrough might trigger such a moment for neuromorphic computing?
What applications would become feasible with such a breakthrough?

If neural networks on neuromorphic computing platforms can gain the accuracy of standard neural networks, the power efficiency gains would represent a tremendous leap forward for broader usage and development of both NN and NM computing.

3. The authors highlight the gap between hardware implementation and software
frameworks in neuromorphic computing compared to traditional deep learning. Develop a
proposal for addressing this gap, specifically focusing on how to create interoperability
between different neuromorphic platforms.

Back to one of the authors' main points, standardization is key. There needs to be a standard API across neuromorphic platforms in order for software tooling to be developed. As with other software, target-specific compilation could be a requirement, but the details should be abstracted from the software developer. It would make sense to require some shared implementation details when first developing cross-compilation tools, such as whether a platform is event-based and whether the signal uses spikes. At some later stage, additional layers of abstraction may be added to allow for common, high-level code to target platforms that differ in these features, but that would not be the initial approach. Similar to CUDA development, eventually there could be tools such as PyTorch that use CUDA but do not require the software developer to have knowledge of the the CUDA backend.

At the high level, there should be some basic classes of objects, such as spiking neurons, which would have standardized methods and configuration parameters. It may be possible to take inspiration from increasingly common async/await coding patterns to model the event-based nature of the neuromorphic computing platforms.

Like with any abstractions, there will be a loss in specificity and power in implementation. For example, there will certainly be configuration options present on some platforms that are not available on others. The API and cross-compilation tool authors will need to decide on reasonable defaults and whether or not there will be a means to customize certain target-specific features.

Perhaps a very basic starting point would be to attempt to wrap existing tooling targeting different platforms, such as PyNN for SpiNNaker and NxSDK / Nengo Loihi for Intel Loihi, with a higher-level implementation. A different approach starting at the other end of the abstraction spectrum would be to create a single cross-compilation tool that uses whatever tooling the aforementioned Python-specific tools use under the hood.

4. The review emphasizes the importance of benchmarks for neuromorphic systems. What
unique metrics would you propose for evaluating neuromorphic systems that go beyond
traditional performance measures like accuracy or throughput? How would you
standardize these across diverse neuromorphic architectures?



5. How might the convergence of emerging memory technologies (like memristors or phase-
change memory) with neuromorphic principles lead to new computational capabilities not
possible with traditional von Neumann architectures? What specific research directions
seem most promising?

## References

[1] D. Kudithipudi, C. Schuman, C. M. Vineyard, et al., “Neuromorphic computing at scale,” *Nature*, vol. 637, pp. 801–812, 2025, doi: 10.1038/s41586-024-08253-8.

[2] C. Teuscher, “Neuromorphic chips,” ECE 510: Hardware for AI and ML, Portland State University, Portland, OR, May 12, 2025, [Lecture].
