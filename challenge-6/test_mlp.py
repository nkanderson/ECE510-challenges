from mlp import MLP
from plot_utils import (
    visualize_hidden_activations,
    plot_mlp_decision_boundary,
    visualize_output_neuron_surface,
    visualize_output_neuron_with_hidden_contributions,
)


def test_mlp_on_xor():
    mlp = MLP()
    xor_inputs = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    for inputs, expected in xor_inputs:
        x1, x2 = inputs
        output = mlp.forward(x1, x2)
        print(f"Input: {inputs}, Output: {output:.4f}, Expected: {expected}")

    visualize_hidden_activations(mlp, name="before-training")


if __name__ == "__main__":
    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    mlp = MLP()
    # This was relatively uninteresting
    # print("Before training:")
    # visualize_hidden_activations(mlp, name="before-training")

    mlp.train(xor_data, epochs=10000, learning_rate=0.5)

    for inputs, target in xor_data:
        output = mlp.forward(*inputs)
        print(f"Input: {inputs}, Output: {output:.4f}, Target: {target}")

    # print("\nAfter training:")
    visualize_hidden_activations(mlp, save_dir="plots")

    plot_mlp_decision_boundary(mlp, save_dir="plots")
    # visualize_output_neuron_surface(mlp)
    visualize_output_neuron_with_hidden_contributions(mlp, save_dir="plots")
