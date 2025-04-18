import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


class MLP:
    def __init__(self):
        # Hidden layer weights and biases
        # Each hidden neuron gets 2 inputs
        self.hidden_weights = [
            [random.uniform(-1, 1), random.uniform(-1, 1)],  # weights for h1
            [random.uniform(-1, 1), random.uniform(-1, 1)],  # weights for h2
        ]
        self.hidden_biases = [random.uniform(-1, 1) for _ in range(2)]

        # Output layer weights (connecting h1, h2) and bias
        self.output_weights = [random.uniform(-1, 1) for _ in range(2)]
        self.output_bias = random.uniform(-1, 1)

        self.snapshots = []  # for animation

    def hidden_outputs(self, x1, x2):
        outputs = []
        for i in range(2):
            w = self.hidden_weights[i]
            b = self.hidden_biases[i]
            z = w[0] * x1 + w[1] * x2 + b
            h = sigmoid(z)
            outputs.append(h)
        return outputs

    def forward(self, x1, x2):
        # Compute hidden activations
        hidden_outputs = []
        for i in range(2):
            w = self.hidden_weights[i]
            b = self.hidden_biases[i]
            z = w[0] * x1 + w[1] * x2 + b
            h = sigmoid(z)
            hidden_outputs.append(h)

        # Compute output neuron activation
        z_out = (
            self.output_weights[0] * hidden_outputs[0]
            + self.output_weights[1] * hidden_outputs[1]
            + self.output_bias
        )
        y = sigmoid(z_out)
        return y

    def train(
        self,
        data,
        epochs=10000,
        learning_rate=0.1,
        log_interval=1000,
        interval_epochs=100,
    ):
        for epoch in range(epochs):
            total_error = 0
            for (x1, x2), target in data:
                ### FORWARD PASS ###
                # Hidden layer
                hidden_z = []
                hidden_out = []
                for i in range(2):
                    z = (
                        self.hidden_weights[i][0] * x1
                        + self.hidden_weights[i][1] * x2
                        + self.hidden_biases[i]
                    )
                    hidden_z.append(z)
                    hidden_out.append(sigmoid(z))

                # Output layer
                z_out = (
                    self.output_weights[0] * hidden_out[0]
                    + self.output_weights[1] * hidden_out[1]
                    + self.output_bias
                )
                y_out = sigmoid(z_out)

                error = target - y_out
                total_error += error**2

                ### BACKPROPAGATION ###
                # Output layer gradients
                delta_out = error * sigmoid_derivative(y_out)
                d_output_weights = [
                    delta_out * hidden_out[0],
                    delta_out * hidden_out[1],
                ]
                d_output_bias = delta_out

                # Hidden layer gradients
                delta_hidden = []
                for i in range(2):
                    delta = (
                        sigmoid_derivative(hidden_out[i])
                        * self.output_weights[i]
                        * delta_out
                    )
                    delta_hidden.append(delta)

                d_hidden_weights = [
                    [delta_hidden[0] * x1, delta_hidden[0] * x2],
                    [delta_hidden[1] * x1, delta_hidden[1] * x2],
                ]
                d_hidden_biases = [delta_hidden[0], delta_hidden[1]]

                ### UPDATE WEIGHTS ###
                for i in range(2):
                    self.output_weights[i] += learning_rate * d_output_weights[i]
                self.output_bias += learning_rate * d_output_bias

                for i in range(2):
                    for j in range(2):
                        self.hidden_weights[i][j] += (
                            learning_rate * d_hidden_weights[i][j]
                        )
                    self.hidden_biases[i] += learning_rate * d_hidden_biases[i]

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}: Total Error = {total_error:.4f}")

            if epoch % interval_epochs == 0:
                self.snapshots.append(
                    {
                        "hidden_weights": [w.copy() for w in self.hidden_weights],
                        "hidden_biases": self.hidden_biases.copy(),
                        "output_weights": self.output_weights.copy(),
                        "output_bias": self.output_bias,
                        "epoch": epoch,
                    }
                )
