# Basic Perceptron Implementation

**Niklas Anderson**  
**April 2025**

## Results

- Asked ChatGPT to step through an example of 3 epochs for the NAND training in order to better understand gradient descent. Got a better sense of how each weight is updated in very small increments
- Asked how the input space is determined to better understand the specific task of learning XOR
- Asked why a single-layer perceptron is like a straight line, and how hidden layers contribute to more complex functionality

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
