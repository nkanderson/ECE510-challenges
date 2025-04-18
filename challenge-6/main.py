import argparse
from perceptron import Perceptron
import matplotlib.pyplot as plt


def train_and_plot(data, name: str, do_plot: bool):
    print(f"\nTraining for {name.upper()}")
    p = Perceptron(2, learning_rate=0.1)
    p.train(data, epochs=10000, log_interval=1000)

    print("\nFinal Predictions:")
    for inputs, target in data:
        pred = p.predict(inputs)
        print(f"Input: {inputs}, Predicted: {pred}, Target: {target}")

    if not do_plot:
        return

    # Plot total error
    plt.figure()
    plt.plot(p.loss_history)
    plt.title(f"Learning Curve ({name.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.grid(True)
    plt.show()

    # Plot weight and bias evolution
    w0 = [w[0] for w in p.weight_history]
    w1 = [w[1] for w in p.weight_history]
    b = p.bias_history

    plt.figure()
    plt.plot(w0, label="Weight 0", linewidth=2, linestyle="-")
    plt.plot(w1, label="Weight 1", linewidth=2, linestyle="--")
    plt.plot(b, label="Bias", linewidth=2, linestyle="-.")
    plt.title(f"Weights and Bias Evolution ({name.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_nand(do_plot: bool):
    nand_data = [
        ([0, 0], 1),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]
    train_and_plot(nand_data, "nand", do_plot)


def train_xor(do_plot: bool):
    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]
    train_and_plot(xor_data, "xor", do_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true", help="Enable plotting of training curves"
    )
    parser.add_argument("--skip-nand", action="store_true", help="Skip NAND training")
    parser.add_argument("--skip-xor", action="store_true", help="Skip XOR training")
    args = parser.parse_args()

    if not args.skip_nand:
        train_nand(args.plot)
    if not args.skip_xor:
        train_xor(args.plot)
