# Basic Perceptron Implementation

**Niklas Anderson**  
**April 2025**

## Results

- Asked ChatGPT to step through an example of 3 epochs for the NAND training in order to better understand gradient descent. Got a better sense of how each weight is updated in very small increments
- Asked how the input space is determined to better understand the specific task of learning XOR
- Asked why a single-layer perceptron is like a straight line, and how hidden layers contribute to more complex functionality
-- From ChatGPT: "The inputs can’t be transformed — they're passed straight into a single weighted sum and a sigmoid."
- On initially training an MLP on XOR, it failed to correctly train on all input combinations. The initial weights and biases had been hardcoded to 0.5 for hidden weights, 0.0 for hidden biases, 1.0 for output weights, and -1.0 for output bias. After adjusting all to be random in the range -1 to 1, it more often than not correctly trained with the same number of epochs.

### Main points from ChatGPT
- A 2-2-1 network means there are 2 input neurons, 2 hidden neurons (in one hidden layer), and 1 output neuron
-- Each neuron in the hidden layer computes its own weighted sum and nonlinear activation. This enables the MLP to transform the input space before making a decision.
- Hidden neurons can create new features or partition space nonlinearly
- The output neuron combines these nonlinear features
- The network can form composite decision boundaries

Describing why a single perceptron is linear while hidden layers may be nonlinear:
Let’s start with the perceptron:
```
output = sigmoid(w1 * x1 + w2 * x2 + b)
```
This is:
- A linear combination of the inputs: z=w1x1+w2x2+bz=w1​x1​+w2​x2​+b
- Then passed through a sigmoid (or other activation)

But crucially — when drawing a decision boundary, you’re checking when the output is constant, like:
```
sigmoid(w1 * x1 + w2 * x2 + b) = 0.5
```
That’s equivalent to:
```
w1 * x1 + w2 * x2 + b = 0  # ← linear boundary
```
This defines a straight line (or hyperplane in higher dimensions).

So even though the sigmoid is nonlinear, the decision boundary (where output = threshold) is linear because it’s just where the weighted sum hits a constant.

Hidden neurons are nonlinear functions of linear combinations. Output neurons are nonlinear functions of hidden outputs. Their composition breaks linearity and gives rise to flexible decision surfaces.

- As noted above, MLP XOR training failed with hardcoded weights. ChatGPT said random initialization "breaks the symmetry and allows hidden neurons to specialize."
-- On digging in a bit more, ChatGPT indicated the 2 hidden neurons get the exact same updates for weights and bias when they start out the same, so they are essentially clones. As a result, the "network can’t learn anything more expressive than a single hidden unit"
- The hidden neuron activations don't necessarily look like a decomposition of an XOR, but each may still identify features that the output function is then able to compose into the correct output.

#### Questions
- What does it mean to check for when the output is constant?
- Is the definition of the decision boundary where output = threshold?
-- Both of the above seem to relate to the actual sigmoid definition itself, as it crosses from negative to positive x values at y = 0.5 (which were the example numbers ChatGPT provided in this explanation). But it's still unclear to me what it means for the output to be constant. Is this related to gradient descent and finding a local minimum using the derivative? That doesn't seem to make sense, because that's for training. What threshold is this referencing?

## Commands

Show decision boundary live:
```sh
python main.py --plots decision-boundary
```

Show and save XOR decision boundary animation:
```sh
python main.py --plots decision-boundary --skip-nand --gif
```

Full training for both, no animation:
```sh
python main.py --plots learning weights
```

Live view of plots for decision boundary, learning curve, and weights and bias, plus saved gifs:
```sh
python main.py --plots decision-boundary --gif --skip-nand
```
