import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from perceptron import Perceptron


def plot_learning_curve(loss_history, name):
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Learning Curve ({name.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weights_and_bias(weight_history, bias_history, name):
    w0 = [w[0] for w in weight_history]
    w1 = [w[1] for w in weight_history]
    b = bias_history

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


def plot_decision_boundary(model: Perceptron, data, name):
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([model.predict(list(point)) for point in grid])
    zz = zz.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, zz, levels=[-1, 0.5, 1], cmap="coolwarm", alpha=0.6)
    plt.title(f"Decision Boundary ({name.upper()})")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")

    for x, label in data:
        plt.scatter(
            x[0],
            x[1],
            c="black" if label == 1 else "white",
            edgecolors="k",
            s=100,
            marker="o",
        )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_hidden_activations(mlp, name="mlp", resolution=200, save_dir=None):
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Plot each hidden neuron's activation
    for i in range(2):
        zz = np.zeros_like(xx)
        for j in range(resolution):
            for k in range(resolution):
                x1 = xx[j, k]
                x2 = yy[j, k]
                h = mlp.hidden_outputs(x1, x2)[i]
                zz[j, k] = h

        plt.figure()
        plt.contourf(xx, yy, zz, levels=50, cmap="viridis")
        plt.colorbar()
        plt.title(f"Hidden Neuron {i+1} Activation ({name})")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/{name}_h{i+1}_activation.png")
            plt.close()
        else:
            plt.show()


def visualize_output_neuron_surface(mlp, name="mlp", resolution=200, save_dir=None):
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    zz = np.zeros_like(xx)

    for i in range(resolution):
        for j in range(resolution):
            x1 = xx[i, j]
            x2 = yy[i, j]
            h1, h2 = mlp.hidden_outputs(x1, x2)
            z_out = (
                mlp.output_weights[0] * h1
                + mlp.output_weights[1] * h2
                + mlp.output_bias
            )
            zz[i, j] = z_out  # not sigmoid! raw output activation

    plt.figure()
    contour = plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.8)
    plt.colorbar(contour, label="Output Neuron Pre-Activation (z)")
    plt.title(f"Output Neuron Pre-Sigmoid Activation ({name})")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # XOR training points
    for (x1, x2), label in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        plt.scatter(x1, x2, c="black" if label == 1 else "white", edgecolors="k", s=100)

    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{name}_output_neuron_presig_activation.png")
        plt.close()
    else:
        plt.show()


def visualize_output_neuron_with_hidden_contributions(
    mlp, name="mlp", resolution=200, save_dir=None
):
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    z_out_vals = np.zeros_like(xx)
    h1_vals = np.zeros_like(xx)
    h2_vals = np.zeros_like(xx)

    for i in range(resolution):
        for j in range(resolution):
            x1 = xx[i, j]
            x2 = yy[i, j]
            h1, h2 = mlp.hidden_outputs(x1, x2)
            h1_vals[i, j] = h1
            h2_vals[i, j] = h2
            z_out_vals[i, j] = (
                mlp.output_weights[0] * h1
                + mlp.output_weights[1] * h2
                + mlp.output_bias
            )

    plt.figure()

    # Main surface: z_out (pre-sigmoid)
    contour = plt.contourf(xx, yy, z_out_vals, levels=50, cmap="RdBu", alpha=0.8)
    plt.colorbar(contour, label="Output Neuron Pre-Activation (z)")

    # Overlay h1 contour lines
    c1 = plt.contour(
        xx, yy, h1_vals, levels=[0.2, 0.5, 0.8], colors="yellow", linestyles="dashed"
    )
    plt.clabel(c1, inline=True, fontsize=8, fmt="h1=%.1f")

    # Overlay h2 contour lines
    c2 = plt.contour(
        xx, yy, h2_vals, levels=[0.2, 0.5, 0.8], colors="lime", linestyles="solid"
    )
    plt.clabel(c2, inline=True, fontsize=8, fmt="h2=%.1f")

    # XOR training points
    for (x1, x2), label in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        plt.scatter(x1, x2, c="black" if label == 1 else "white", edgecolors="k", s=100)

    plt.title(f"Output Neuron + Hidden Contributions ({name})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{name}_output_neuron_with_hidden.png")
        plt.close()
    else:
        plt.show()


def plot_mlp_decision_boundary(mlp, name="mlp", resolution=200, save_dir=None):
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    zz = np.zeros_like(xx)
    for i in range(resolution):
        for j in range(resolution):
            zz[i, j] = mlp.forward(xx[i, j], yy[i, j])

    plt.figure()
    plt.contourf(xx, yy, zz, levels=[-1, 0.5, 1], cmap="coolwarm", alpha=0.6)
    plt.colorbar()
    plt.title(f"Decision Boundary ({name})")
    plt.xlabel("x1")
    plt.ylabel("x2")

    for (x1, x2), label in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        plt.scatter(x1, x2, c="black" if label == 1 else "white", edgecolors="k", s=100)

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{name}_decision_boundary.png")
        plt.close()
    else:
        plt.show()


def animate_decision_boundary(
    weight_history, bias_history, data, name, interval_epochs=100, save_gif=False
):
    fig, ax = plt.subplots()
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.zeros_like(xx)

    def update(frame):
        ax.clear()
        weights = weight_history[frame]
        bias = bias_history[frame]
        Z = np.array(
            [
                1 / (1 + np.exp(-(weights[0] * x + weights[1] * y + bias)))
                for x, y in grid
            ]
        )
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1, 0.5, 1], cmap="coolwarm", alpha=0.6)
        ax.set_title(
            f"{name.upper()} Decision Boundary (Epoch {frame * interval_epochs})"
        )
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        for x, label in data:
            ax.scatter(
                x[0],
                x[1],
                c="black" if label == 1 else "white",
                edgecolors="k",
                s=100,
                marker="o",
            )

    frames = list(range(0, len(weight_history), interval_epochs))
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=200, repeat=False
    )

    if save_gif:
        gif_name = f"{name}_decision_boundary.gif"
        print(f"Saving animation to {gif_name}...")
        ani.save(gif_name, writer="pillow")

    plt.show()


def animate_learning_curve(loss_history, name, interval_epochs=100, save_gif=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(loss_history))
    ax.set_ylim(0, max(loss_history) * 1.1)
    (line,) = ax.plot([], [], lw=2)

    def update(frame):
        x = list(range(frame * interval_epochs))
        y = loss_history[: frame * interval_epochs]
        line.set_data(x, y)
        ax.set_title(f"{name.upper()} Learning Curve (Epoch {frame * interval_epochs})")
        return (line,)

    frames = list(range(1, len(loss_history) // interval_epochs))
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)

    if save_gif:
        gif_name = f"{name}_learning_curve.gif"
        print(f"Saving learning curve GIF to {gif_name}...")
        ani.save(gif_name, writer="pillow")

    plt.show()


def animate_weights_and_bias(
    weight_history, bias_history, name, interval_epochs=100, save_gif=False
):
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(weight_history))
    all_values = (
        [w[0] for w in weight_history] + [w[1] for w in weight_history] + bias_history
    )
    ax.set_ylim(min(all_values) - 0.1, max(all_values) + 0.1)

    (w0_line,) = ax.plot([], [], label="Weight 0", lw=2, linestyle="-")
    (w1_line,) = ax.plot([], [], label="Weight 1", lw=2, linestyle="--")
    (b_line,) = ax.plot([], [], label="Bias", lw=2, linestyle="-.")

    def update(frame):
        end = frame * interval_epochs
        x = list(range(end))
        w0_vals = [w[0] for w in weight_history[:end]]
        w1_vals = [w[1] for w in weight_history[:end]]
        b_vals = bias_history[:end]

        w0_line.set_data(x, w0_vals)
        w1_line.set_data(x, w1_vals)
        b_line.set_data(x, b_vals)

        ax.set_title(f"{name.upper()} Weights/Bias Evolution (Epoch {end})")
        return w0_line, w1_line, b_line

    frames = list(range(1, len(weight_history) // interval_epochs))
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)

    ax.legend()
    ax.grid(True)

    if save_gif:
        gif_name = f"{name}_weights_bias.gif"
        print(f"Saving weight/bias evolution GIF to {gif_name}...")
        ani.save(gif_name, writer="pillow")

    plt.show()
