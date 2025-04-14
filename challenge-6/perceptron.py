import math
import random


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    return y * (1 - y)


class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.lr = learning_rate

    def forward(self, inputs: list[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return sigmoid(z)

    def train(self, training_data: list[tuple[list[int], int]], epochs: int = 10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                delta = error * sigmoid_derivative(output)
                # Update weights and bias
                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * delta * inputs[i]
                self.bias += self.lr * delta
                total_error += error**2
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Total Error = {total_error:.4f}")

    def predict(self, inputs: list[int]) -> int:
        return round(self.forward(inputs))
