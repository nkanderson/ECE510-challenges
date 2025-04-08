# Intrinsic and Designed Computation

**Niklas Anderson**  
**April 2025**

## Intrinsic Computation Overview

The authors in [1] provide a high-level overview of intrinsic computation, an emergent characteristic of dynamical systems related to their information storage and processing which produces resultant behavior and organization. This form of computation is in contrast with designed computation, which has some utility or useful behavior for its designers. They discuss historical work related to chaotic dynamical systems, organized complexity, cybernetics, and information theory, among other topics.

## Information and Communication Theory

Authors of [1] note that Claude Shannon's information theory has been used widely in assessing information processing, but his understanding of communication theory has much more potential application. They state that his work related to cryptographic systems is similarly underutilized. Specifically, they state that "there has been relatively little progress in analyzing the architecture of how complex systems support the flow and storage of information." This investigation may provide important insights for addressing well-known challenges in modern Machine Learning (ML) related to data processing and transfer. The authors of [1] do not explore the potential application of Shannon's work on communication theory in depth, simply noting it as an area for future research.

## Intrinsic Computation and Artificial Intelligence

Though the authors of [1] were primarily focused on alternative substrates for computation and the development of alternative models for designed computation inspired by examples of intrinsic computation, it is interesting to consider intrinsic computation as it relates to the current state of Artifical Intelligence (AI). At the time of publication of [1] in 2010, the field of AI was in a relative dormant period, just prior to renewed breakthroughs in deep learning, per [2]. Understandably, the authors of [1] expressed skepticism about the prospects for AI. Since then, [2] has noted there have been significant gains, and a Third Golden Age for AI with deep learning has taken place. I believe the current state of AI with deep learning in some applications is an effort to model the instrinsic computation of dynamical systems as described by [1].

Consider a weather pattern as an example of instrinsic computation - it is a dynamical system with a great deal of information which results in behavior or output. Current ML models may be used to process relevant physical factors to predict this dynamical system's output. AI can be thought of more generally as an attempt to predict rather than produce the output of a dynamical system. ChatGPT, for example, is the result of consumption and processing of massive amounts of data, with the end goal being the approximation of a human understanding of the world with the breadth and depth of knowledge no single human can possess. ML is, in other words, a process of intrinsic computation discovery.

## Related Research

A number of the articles described in [1] seem potentially quite relevant to current discussions on Machine Learning (ML), including the following:
- _Intrinsic Information Carriers in Combinatorial Dynamical Systems_: The "graph-based framework of rewrite rules" described by the authors in [1] seem related to the trained weights in ML models. The demonstration of "aggregated variables that reflect the causal structure laid down by the mechanisms expressed by the rules" are interesting to consider in light of attempts to reduce the data consumption of ML models during training.
- _Optimal Causal Inference: Estimating Stored Information and Approximating Causal Architecture_: Authors of [1] describe the "approach to inferring the causal architecture of a stochastic dynamical system" from this paper, as well as the "quantitative way of exploring fundamental tradeoffs associated with model complexity versus model predictability." These components both seem relevant to ML model training and reduction of data consumption.
- _Computing Adaptive Bayesian Inference from Multiple Sources_: As described in [1], this paper directly discusses neural systems and the calculation of posterior probabilities and learning relative reliabilities with a single algorithm.

The above articles may be interesting further reading during investigation of how research on instrinsic computation is relevant to modern AI and ML. Additionally, application of Shannon's communication theory as it relates to data flow and storage within complex systems could be a good starting point in understanding current techniques for managing data flow and storage in ML training and inference.

## References

[1] J. P. Crutchfield, W. L. Ditto, and S. Sinha, “Introduction to Focus Issue: Intrinsic and Designed Computation: Information Processing in Dynamical Systems—Beyond the Digital Hegemony,” Chaos, vol. 20, no. 3, p. 037101, Sep. 2010, doi: 10.1063/1.3492712.

[2] C. Teuscher, “Hardware for AI and ML Overview + Co-design,” ECE 510: Hardware for AI and ML, Portland State University, Portland, OR, Apr. 7, 2025, [Lecture].
