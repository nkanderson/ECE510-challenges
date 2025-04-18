# Basic Perceptron Implementation

**Niklas Anderson**  
**April 2025**

## Results

- Asked ChatGPT to step through an example of 3 epochs for the NAND training in order to better understand gradient descent. Got a better sense of how each weight is updated in very small increments
- Asked how the input space is determined to better understand the specific task of learning XOR
- Asked why a single-layer perceptron is like a straight line, and how hidden layers contribute to more complex functionality
-- From ChatGPT: "The inputs can’t be transformed — they're passed straight into a single weighted sum and a sigmoid."

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
