import argparse
from perceptron import Perceptron
from plot_utils import (
    plot_learning_curve,
    plot_weights_and_bias,
    plot_decision_boundary,
    animate_decision_boundary,
    animate_learning_curve,
    animate_weights_and_bias,
)


def train_and_plot(data, name: str, plots: list[str], gif: bool):
    print(f"\nTraining for {name.upper()}")
    p = Perceptron(2, learning_rate=0.1)
    p.train(data, epochs=10000, log_interval=1000)

    print("\nFinal Predictions:")
    for inputs, target in data:
        pred = p.predict(inputs)
        print(f"Input: {inputs}, Predicted: {pred}, Target: {target}")

    if "learning" in plots:
        plot_learning_curve(p.loss_history, name)

    if "weights" in plots:
        plot_weights_and_bias(p.weight_history, p.bias_history, name)

    if "decision-boundary" in plots:
        plot_decision_boundary(p, data, name)

    if gif:
        animate_decision_boundary(
            weight_history=p.weight_history,
            bias_history=p.bias_history,
            data=data,
            name=name,
            interval_epochs=100,
            save_gif=True,
        )

        animate_learning_curve(
            loss_history=p.loss_history,
            name=name,
            interval_epochs=100,
            save_gif=True,
        )

        animate_weights_and_bias(
            weight_history=p.weight_history,
            bias_history=p.bias_history,
            name=name,
            interval_epochs=100,
            save_gif=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots",
        nargs="+",
        default=[],
        help="Specify which plots to generate: learning, weights, decision-boundary",
    )
    parser.add_argument("--gif", action="store_true", help="Generate animation as GIF")
    parser.add_argument("--skip-nand", action="store_true", help="Skip NAND training")
    parser.add_argument("--skip-xor", action="store_true", help="Skip XOR training")
    args = parser.parse_args()

    if not args.skip_nand:
        nand_data = [
            ([0, 0], 1),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]
        train_and_plot(nand_data, "nand", args.plots, args.gif)

    if not args.skip_xor:
        xor_data = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]
        train_and_plot(xor_data, "xor", args.plots, args.gif)


if __name__ == "__main__":
    main()
