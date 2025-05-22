# Paper Review: *Neuromorphic computing at scale*

**Niklas Anderson**  
**May 2025**

## Neuromorphic Computation Software Frameworks: Creating interoperability between different neuromorphic platforms

A widely acknowledged challenge facing widespread adoption of neuromorphic computing architectures is the lack of a robust and standardized software ecosystem [1][2][3]. There are a number of quality tools, such as [Nengo](https://www.nengo.ai/) and [Lava](https://lava-nc.org/index.html) which provide support at different levels for development of applications on neuromorphic hardware. However, there appears to be a lack of a dominant framework that falls into the role of providing a vendor-agnostic, neuromorphic-only framework. The following will lay out the need for such a framework and outline characteristics required to make it a beneficial new addition, rather than a duplicative effort in an already fragmented ecosystem.

Nengo is a high-quality framework that covers a wide range of possible use cases, with what seems to be a "batteries-included" approach. Its focus appears to be on implementation and usage of neural network algorithms, allowing for execution on different backing hardware, which may include neuromorphic compute options. This is a clear win for many users. However, this approach necessitates a higher level of abstraction than would be required with a framework more targeted to neuromorphic hardware. As with any abstraction, some degree of control, power, and efficiency is often lost, in exchange for ease-of-use. This is the appropriate tradeoff in many cases, but not all.

Lava is an example that exists at the other end of the abstraction spectrum, with regard to vendor specificity. It is an open source project, which is likely to produce a solution which has broad applicability to a range of use cases. However, it only targets Intel's Loihi 2 neuromorphic chip.

A framework that exists in between these two, with a focus on neuromorphic chips only but support for multiple vendor backends, would allow for the strengths of neuromorphic chips to be fully leveraged while allowing for portability and a decreased time-to-value for users creating their first neuromorphic compute application. As noted above, the high level of abstraction provided by Nengo is perfect for many users who want to implement a neural network on any of a number of backends. But the abstraction requires some loss of fidelity to the backend specifics. In other words, the end user does not have as much control over how their code ultimately executes. For users who want to leverage the strengths of neuromorphic computing for the first time, being able to avoid complete vendor lock-in is a huge benefit.

A framework that allows for a neuromorphic-specific abstraction across vendors requires some of the following characteristics in order to be successful:

**Standardized integration with existing solutions**  
Authors in [1] and [2] explain in depth why standardization is essential for increased adoption of neuromorphic computing. Their main point is well taken, that outside of research, most new usage of neuromorphic computing will need to work within heterogeneous systems. This points to the need to common APIs between systems. It may be worth exploring the APIs developed by Nengo for inspiration.

**Multiple supported vendor backends**  
This is implied in such a framework's description as vendor-agnostic, but it's worth noting how this may be accomplished. Initially there may be 2-3 of the most popular backends chosen to develop integrations for. These may shared characteristics to ease initial development, such as whether they are event-based and whether the signals use spikes[3]. It should be possible for community-developed extensions to work in place of these integrations, allowing for expansion of supported backends over time. It would be worth examining existing vendor-specific frameworks such as Lava in order to understand the libraries and work required to develop target-specific compilation tools.

**Configurable, but opinionated**  
The framework should follow common best practices within the language for application configuration, making it straightforward to set up a new project with user-specified characteristics. In order to ease the onboarding for users new to neuromorphic computing, reasonable defaults and design patterns should be chosen.

**Intuitive model for neuromorphic primitives**  
It may be worth reviewing existing vendor-specific libraries to understand whether there are common representations for fundamental neuromorphic compute components. The framework should avoid vendor-specific functionality in the most basic object models, but allow for extension or addition of vendor-specific libraries where desired. The basic objects should have standard methods and configuration parameters that would be shared across vendors.

With the above goals in mind, it may be possible to create a framework which would expand adoption of neuromorphic computing. Such an effort would rely on a certain degree of openness on behalf of vendors, but this is a tradeoff that has been proven effective across many software and hardware projects. This framework would draw inspiration from and have overlapping goals with existing frameworks such as Nengo, but also fulfill a unique need that must be met in order to grow a healthy ecosystem catering to a range of users.


## References

[1] D. Kudithipudi, C. Schuman, C. M. Vineyard, et al., “Neuromorphic computing at scale,” *Nature*, vol. 637, pp. 801–812, 2025, doi: 10.1038/s41586-024-08253-8.

[2] D. R. Muir and S. Sheik, “The road to commercial success for neuromorphic technologies,” *Nature Communications*, vol. 16, no. 1, p. 3586, Apr. 2025, doi: 10.1038/s41467-025-57352-1.

[3] C. Teuscher, “Neuromorphic chips,” ECE 510: Hardware for AI and ML, Portland State University, Portland, OR, May 12, 2025, [Lecture].

## Appendix: Original Questions and Notes

1. The authors discuss several key features necessary for neuromorphic systems at scale (distributed hierarchy, sparsity, neuronal scalability, etc.). Which of these features do you believe presents the most significant research challenge, and why? How might overcoming this challenge transform the field?

Challenges in implementing distributed hierarchy include modularization of different components, as well as establishing a standardized communication protocol between components. It would likely require understanding the minimal data representation necessary for each component to produce and pass to other components. In other words, it would require establishing APIs between all the components. As noted, working towards this goal would increase explainability, which is a challenge in current AI models. Modularization has typically supported greater adoption, as it allows for specialization and abstraction.

Plasticity or dynamic reconfigurability is described by the authors as being in an early stage of maturity. They define requirements for this feature as being rooted in patterns seen in human cognition, including networks "that interact in complex and transient communication patterns"[1]. The huge design space for this particular feature seems to add to the challenges in its research. To even begin research, it is necessary to choose a particular focus, resulting in the exclusion of many other possible avenues. For example, at what level of plasticity would it make sense to begin with - the individual neurons, or the system as a whole? The authors note it may be either. If one were to choose the individual neurons, there is then the challenge of analyzing the massive number of possible combinations of neuron states as they interact with one another. It seems like the secondary challenge is that of analyzing combinatorial states, after the primary challenge of identifying what variable conditions to allow.

The power of neuronal plasticity seems apparent, if the challenges could be overcome. For example, it seems like having reconfigurable neurons could support other features such as heterogeneous functionality and hierarchical systems, though the ability to modify neuronal activity depending on the context. If reconfigurable neurons were generally available and had a straightforward API, it's easy to imagine much broader adoption, as deployment and usage would have greater flexibility.

2. The article compares neuromorphic computing's development to the evolution of deep learning, suggesting it awaits its own "AlexNet moment." What specific technological or algorithmic breakthrough might trigger such a moment for neuromorphic computing? What applications would become feasible with such a breakthrough?

If neural networks on neuromorphic computing platforms can gain the accuracy of standard neural networks, the power efficiency gains would represent a tremendous leap forward for broader usage and development of both NN and NM computing.

3. The authors highlight the gap between hardware implementation and software frameworks in neuromorphic computing compared to traditional deep learning. Develop a proposal for addressing this gap, specifically focusing on how to create interoperability between different neuromorphic platforms.

Back to one of the authors' main points, standardization is key. There needs to be a standard API across neuromorphic platforms in order for software tooling to be developed. As with other software, target-specific compilation could be a requirement, but the details should be abstracted from the software developer. It would make sense to require some shared implementation details when first developing cross-compilation tools, such as whether a platform is event-based and whether the signal uses spikes. At some later stage, additional layers of abstraction may be added to allow for common, high-level code to target platforms that differ in these features, but that would not be the initial approach. Similar to CUDA development, eventually there could be tools such as PyTorch that use CUDA but do not require the software developer to have knowledge of the the CUDA backend.

At the high level, there should be some basic classes of objects, such as spiking neurons, which would have standardized methods and configuration parameters. It may be possible to take inspiration from increasingly common async/await coding patterns to model the event-based nature of the neuromorphic computing platforms.

Like with any abstractions, there will be a loss in specificity and power in implementation. For example, there will certainly be configuration options present on some platforms that are not available on others. The API and cross-compilation tool authors will need to decide on reasonable defaults and whether or not there will be a means to customize certain target-specific features.

Perhaps a very basic starting point would be to attempt to wrap existing tooling targeting different platforms, such as PyNN for SpiNNaker and NxSDK / Nengo Loihi for Intel Loihi, with a higher-level implementation. A different approach starting at the other end of the abstraction spectrum would be to create a single cross-compilation tool that uses whatever tooling the aforementioned Python-specific tools use under the hood.

4. The review emphasizes the importance of benchmarks for neuromorphic systems. What unique metrics would you propose for evaluating neuromorphic systems that go beyond traditional performance measures like accuracy or throughput? How would you standardize these across diverse neuromorphic architectures?



5. How might the convergence of emerging memory technologies (like memristors or phase-change memory) with neuromorphic principles lead to new computational capabilities not possible with traditional von Neumann architectures? What specific research directions seem most promising?
