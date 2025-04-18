# LLM Transcript

## ChatGPT 4o

Prompt: I'd like to implement a simple neuron (perceptron) with 2 inputs and a sigmoid activation function in Python 3.12. It will use the perceptron learning rule to train the neuron for the binary logic functions NAND and XOR. I want a brief explanation of the perceptron learning rule before we begin. If it's simpler, it's ok to separate the training for each binary function into two difference stages.

Response: Sure! Let's start with a brief explanation of the Perceptron Learning Rule, then we can move on to implementing it.

🔹 Perceptron Learning Rule (with sigmoid activation)

The perceptron learning rule is a simple algorithm for updating the weights of a neuron to reduce the error between the predicted output and the desired output (label).

For a perceptron with:

    inputs: x1,x2,...,xnx1​,x2​,...,xn​

    weights: w1,w2,...,wnw1​,w2​,...,wn​

    output: y=activation(w⋅x+b)y=activation(w⋅x+b)

The learning rule is:
wi←wi+η⋅(d−y)⋅xi
wi​←wi​+η⋅(d−y)⋅xi​
b←b+η⋅(d−y)
b←b+η⋅(d−y)

Where:

    ηη: learning rate

    dd: desired output

    yy: predicted output

    xixi​: input

    wiwi​: weight

    bb: bias

Because we're using a sigmoid activation function, which is differentiable, we can improve this rule using the gradient of the sigmoid:
δ=(d−y)⋅y⋅(1−y)
δ=(d−y)⋅y⋅(1−y)
wi←wi+η⋅δ⋅xi
wi​←wi​+η⋅δ⋅xi​
b←b+η⋅δ
b←b+η⋅δ

This is effectively gradient descent on the error using the sigmoid function.

Next up, I’ll write a Python class for a perceptron with:

    Two inputs

    Sigmoid activation

    Training method using the rule above

    Support for training on NAND and XOR separately

Want to dive into the code?

Prompt: Yes

Response: Great! Here's a simple implementation of a Perceptron with sigmoid activation in Python 3.12, including training support for both NAND and XOR logic gates.

<added code to perceptron.py and main.py, had to manually compose and add the import statements to main.py, ChatGPT provided standalone functions for the training>

🔸 Notes

    NAND should converge well — it’s linearly separable.

    XOR won’t converge with a single-layer perceptron because it's not linearly separable. You’d need a multi-layer network (i.e., MLP) to learn XOR properly — but it's still fun to see how close it gets.

Want me to show you a working XOR solution with a 2-layer neural network next?

Prompt: First let's dive a little deeper into the perceptron class code. It looks like the forward method includes a matrix multiple and accumulation, is that correct? And it looks like the train method is using the perceptron learning rule, or basically as you described it, gradient descent. I understand that that's what's happening, but can you describe the logic, specifically behind this snippet:

for i in range(len(self.weights)):
                    self.weights[i] += self.lr * delta * inputs[i]
                self.bias += self.lr * delta


Lastly, can you discuss how the number of epochs is determined? I understand that gradient descent may only find a local minimum for the error rather than the true minimum. Does increasing the number of epochs make it more likely to discover the true minimum, or does it just allow for a smaller learning rate, decreasing the error in finding a local minimum? Or something else?

Response: Awesome questions — you're digging into the mechanics and theory behind gradient descent and perceptron training. Let’s break this down step-by-step.

✅ Part 1: forward() = Weighted Sum + Activation

You're absolutely right:

```python
z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
return sigmoid(z)
```

This performs:

    Matrix multiplication (dot product of weights and inputs)

    Bias addition

    Sigmoid activation

In vector notation:
y=σ(w1x1+w2x2+b)
y=σ(w1​x1​+w2​x2​+b)

So yes — it’s equivalent to one neuron (or perceptron) doing:

    Linear combination → w⋅x+bw⋅x+b

    Nonlinear transformation → σ(z)σ(z)

✅ Part 2: Weight & Bias Update Logic

You’re asking about:

```python
for i in range(len(self.weights)):
    self.weights[i] += self.lr * delta * inputs[i]
self.bias += self.lr * delta
```

Let’s interpret this line-by-line.
💡 Variables involved:

    self.lr: learning rate ηη

    delta: error gradient = (d−y)⋅σ′(y)(d−y)⋅σ′(y)

    inputs[i]: input value xixi​

🔍 The math behind the update:

We’re trying to reduce the loss function E=12(d−y)2E=21​(d−y)2, so we compute the gradient of error ∂E/∂wi∂E/∂wi​ and move against it (gradient descent).

Using the chain rule:
∂E∂wi=∂E∂y⋅∂y∂z⋅∂z∂wi=−(d−y)⋅y(1−y)⋅xi
∂wi​∂E​=∂y∂E​⋅∂z∂y​⋅∂wi​∂z​=−(d−y)⋅y(1−y)⋅xi​

Then we negate this and multiply by ηη to get the weight update:
wi←wi+η⋅(d−y)⋅y(1−y)⋅xi=wi+η⋅δ⋅xi
wi​←wi​+η⋅(d−y)⋅y(1−y)⋅xi​=wi​+η⋅δ⋅xi​

And similarly for the bias:
b←b+η⋅δ
b←b+η⋅δ

So this snippet is performing gradient descent on each weight and the bias, using the error signal propagated from the sigmoid activation.

✅ Part 3: Number of Epochs & Convergence

Great question about epochs.
🕰 What is an epoch?

An epoch is one full pass through the training dataset.
🔄 Why do we run multiple epochs?

Because one pass isn’t enough. In each epoch, the weights are nudged slightly in the direction that reduces error. Repeating this multiple times gives the perceptron more chances to converge.

🎯 How do we choose the number of epochs?

There’s no universal rule, but it’s often based on:

    Size & complexity of dataset

    Learning rate

    Whether the training error plateaus

    Desired accuracy or loss threshold

You can:

    Use a fixed number (e.g., 10,000)

    Or implement early stopping (stop when error is low enough or improvement stalls)

🧠 Does more training guarantee a better solution?

Only partially.

    For linearly separable problems (like NAND): yes, more training typically leads to convergence.

    For non-linearly separable problems (like XOR): no amount of training will fix it — the model is fundamentally underpowered (single-layer can’t learn XOR).

    For gradient descent, it typically finds a local minimum, and more epochs help you get closer to it, especially with a small learning rate.

But:

    Too many epochs + too high learning rate = oscillation or divergence

    Too many epochs + too small learning rate = long training, slow improvement

🔁 Summary of Epoch Effects

Parameter	Effect
More epochs	Better convergence, up to a point
Fewer epochs	Faster training, possibly underfit
Small learning rate	More epochs needed, but finer tuning
Large learning rate	Faster but risk overshooting
Stuck in local min	More epochs don’t help; need better model

Let me know if you want to visualize how the weights evolve, or see the learning curve plotted — it's really helpful for intuitions!

Prompt: Yes, I would like to visualize how the weights evolve, and to see the learning curve plotted. First can we step through an example of the first 3 epochs for the NAND training? I also have a follow-up question about epochs and convergence - since initial weights and bias are random, what do you call it when you perform multiple sets of epochs with different random initial weights and biases in order to attempt to find the true minimum error?

Reponse: Perfect — let’s step through the first few training epochs for NAND and show:

    How the weights and bias evolve

    The corresponding total error

Then I’ll explain what it’s called when you restart training with new random weights to find a better solution.

🔂 Step-by-Step: First 3 Epochs for NAND

Let’s say our initial perceptron state is:

```python
weights = [0.5, -0.3]    # Random
bias = 0.1               # Random
learning_rate = 0.1
```

✅ NAND Truth Table

Inputs	Target
[0, 0]	1
[0, 1]	1
[1, 0]	1
[1, 1]	0

🧠 Epoch 1

We loop over each training sample:
1. Input [0, 0], Target = 1

    z=0.5⋅0+(−0.3)⋅0+0.1=0.1z=0.5⋅0+(−0.3)⋅0+0.1=0.1

    y=σ(0.1)≈0.52498y=σ(0.1)≈0.52498

    error=1−0.52498=0.47502error=1−0.52498=0.47502

    δ=0.47502⋅0.52498⋅(1−0.52498)≈0.118δ=0.47502⋅0.52498⋅(1−0.52498)≈0.118

    Weight updates:

        No change: x0=0,x1=0x0​=0,x1​=0

    Bias update:

        b←0.1+0.1⋅0.118=0.1118b←0.1+0.1⋅0.118=0.1118

2. Input [0, 1], Target = 1

    z=0.5⋅0+(−0.3)⋅1+0.1118=−0.1882z=0.5⋅0+(−0.3)⋅1+0.1118=−0.1882

    y=σ(−0.1882)≈0.4531y=σ(−0.1882)≈0.4531

    δ≈0.1359δ≈0.1359

    Update:

        w1←−0.3+0.1⋅0.1359⋅1=−0.2864w1​←−0.3+0.1⋅0.1359⋅1=−0.2864

        b←0.1118+0.1⋅0.1359=0.1254b←0.1118+0.1⋅0.1359=0.1254

3. Input [1, 0], Target = 1

    z=0.5⋅1+(−0.2864)⋅0+0.1254=0.6254z=0.5⋅1+(−0.2864)⋅0+0.1254=0.6254

    y=σ(0.6254)≈0.6515y=σ(0.6254)≈0.6515

    δ≈0.0787δ≈0.0787

    Update:

        w0←0.5+0.1⋅0.0787⋅1=0.5079w0​←0.5+0.1⋅0.0787⋅1=0.5079

        b←0.1254+0.1⋅0.0787=0.1333b←0.1254+0.1⋅0.0787=0.1333

4. Input [1, 1], Target = 0

    z=0.5079+(−0.2864)+0.1333=0.3548z=0.5079+(−0.2864)+0.1333=0.3548

    y≈σ(0.3548)=0.5877y≈σ(0.3548)=0.5877

    δ=−0.1422δ=−0.1422

    Update:

        w0←0.5079−0.0142=0.4937w0​←0.5079−0.0142=0.4937

        w1←−0.2864−0.0142=−0.3006w1​←−0.2864−0.0142=−0.3006

        b←0.1333−0.0142=0.1191b←0.1333−0.0142=0.1191

✅ After Epoch 1:

```python
weights ≈ [0.4937, -0.3006]
bias ≈ 0.1191
total error ≈ (0.475^2 + 0.547^2 + 0.348^2 + 0.5877^2) ≈ 0.23 + 0.30 + 0.12 + 0.35 = 1.0 (approx)
```

Repeat the same logic for Epochs 2 and 3 — the weights will gradually shift toward better classification.

Want me to code up a logger to print exact weight evolution across the first N epochs for visualization?

🔁 Follow-up: Restarting Training from Random Weights

Your intuition is spot on — if we run training multiple times with different random initializations, we’re attempting to escape poor local minima or saddle points.

This process is called:
🔹 Random Restarts (or Multiple Initializations)

Also known as:

    Weight initialization search

    Stochastic restarts

    In modern neural networks: hyperparameter search or ensemble training

It’s commonly used with:

    Non-convex models like neural networks

    Problems where training outcome depends on initial weights

    Models that may converge poorly with bad initial guesses

Sometimes people select the "best run" based on:

    Lowest validation error

    Highest accuracy

    Best convergence behavior

You can even:

    Average the weights

    Use ensemble predictions from multiple trained models

Would you like to see a plot of:

    Training loss vs epoch (learning curve)

    Weight 1 and weight 2 evolution over time

I can add a version of the Perceptron class that logs and plots those.

Prompt: Yes, let's work on creating both of those plots. The logging though should probably only print some portion of the epochs, since 10000, for example, would be more than I want to see.

Response: Perfect — let’s build this in steps:

    Modify the Perceptron class to:

        Record weights and bias across epochs

        Record total error (loss)

    Log periodically, e.g., every 500 or 1000 epochs

    Plot:

        Total error (learning curve)

        Weight evolution (per weight and bias)

✅ Updated Perceptron Class with Logging and History

<provided updated percepton.py file>

✅ Training and Plotting

<provided train_and_plot_nand() function>

🔍 What You’ll See

    A smoothly decreasing error curve if training is effective

    Line plots showing how each weight and the bias are gradually updated over epochs

This is a great way to visualize convergence and whether learning is stable.

Let me know if you'd like the same treatment for XOR (not expected to converge, but still interesting!) or want to try multiple restarts and overlay plots.

Prompt: I will want to do the same for xor, and I'd actually like to update the existing main.py to have the plotting functionality as an option. So for both training on nand and on xor, there should be the option of whether or not to plot it. The following is the existing main.py, for reference:

<pasted in existing main.py>

Response: Perfect — let’s refactor main.py to:

    Accept a flag to enable or disable plotting

    Use a shared train_and_plot function for both NAND and XOR

    Keep the existing terminal output (inputs, predicted, expected)

✅ Updated main.py with Plotting Option

<provided new main.py>

🏃‍♂️ Example Usages

    Run both and plot:

```sh
python main.py --plot
```

Run just XOR (with plots):
```sh
python main.py --plot --skip-nand
```

Run both with no plots:
```sh
python main.py
```

Let me know if you’d like to also:

    Save plots to files instead of showing them

    Add CLI options to configure learning rate or number of epochs

    Overlay multiple training runs for random restarts

Would you like to work on any of those next?

Prompt: Let's fine-tune what we have for a moment before moving on. I like the weights and bias evolution plots, but it's difficult to see when 2 of the values overlap. Are there any adjustments we can make to ensure all lines are visible? I'd also like to better understand why XOR can't converge with a single layer as we have here

Response: Great instincts — let's address both of those points thoroughly.

🔧 Part 1: Improving Visibility of Overlapping Lines in Weight/Bias Plot
Problem:

Sometimes two lines (e.g., weight and bias) are too close or overlapping, making it hard to distinguish them.
✅ Solutions:

    Thicker lines + markers: Help visually separate each series

    Distinct line styles (solid, dashed, dotted)

    Color distinction (automatic with matplotlib)

    Plot in separate subplots (optional but cleaner)

🔄 Updated Plot Code (inside train_and_plot):

Replace the second plot section with this:

<updated weight and bias evolution plotting>

This makes the lines:

    Thicker

    Visually distinct

    Still on the same plot for easy comparison

Let me know if you’d prefer separate plots per line for even more clarity.

🧠 Part 2: Why XOR Can’t Converge with a Single-Layer Perceptron
🚫 Linear Separability: The Core Issue

A single-layer perceptron can only learn functions that are linearly separable — meaning: there’s a straight line (or plane) that can separate the input space into two classes.

Let’s look at XOR:

Input	Output
(0, 0)	0
(0, 1)	1
(1, 0)	1
(1, 1)	0

Visualizing the outputs in the 2D input space:

```
(0,0) → 0    (0,1) → 1

(1,0) → 1    (1,1) → 0
```

You’ll find:

    The 1’s are diagonal

    No straight line can cleanly separate 1’s from 0’s

🔍 This is the textbook example of a non-linearly separable problem.

🤖 What You Need Instead

To learn XOR correctly, you need:

    A nonlinear model

    Specifically, a Multi-Layer Perceptron (MLP) with at least:

        One hidden layer

        Nonlinear activation (e.g. sigmoid, ReLU)

A simple 2-2-1 network can learn XOR:

    2 inputs → 2 hidden units → 1 output

    The hidden layer creates nonlinear boundaries, allowing the network to simulate XOR logic

🔁 Analogy:

Think of a single-layer perceptron as drawing a straight line on paper.

XOR requires drawing a curve or combination of lines. You just can’t do that with a ruler alone.

Let me know if you want to:

    Implement a simple 2-layer neural network for XOR

    Visualize decision boundaries

    Compare training error vs test error for more complex inputs

Or just continue tuning what we’ve got!

Prompt: Visualizing decision boundaries sounds like it might be helpful, but I'd also like to understand more about the rules for organizing the input space. Is it the case that each input needs to map to an axis? For example, in the visualization you provided above, the first input seems to correspond to the y-axis (it only changes when moving vertically) and the second input seems to correspond to the x-axis (it changes when moving horizontally). Is this the correct way to think about it?

I'd also like to understand how a MLP with at least one hidden layer allows for something beyond what you described as the "straight line on paper" of a single-layer perceptron. Lastly, to clarify, you said a 2-2-1 network can learn XOR, which has 2 hidden units - is it one or two hidden layers that are required to learn XOR?

Response: These are excellent questions — you're zeroing in on the geometric intuition behind neural networks, which is key to really understanding what makes them tick. Let’s break your questions down and clarify everything step-by-step.

📊 Part 1: Input Space and Axes

    "Is it the case that each input maps to an axis?"

Yes — that’s exactly right.

When visualizing binary inputs (like for logic gates), we’re placing each input on its own axis:

    Input 1 → x-axis

    Input 2 → y-axis

So we visualize a 2D grid like this:

```
   y
   ↑
 1 | (0,1)  ●──1     (1,1)  ●──0
   |
 0 | (0,0)  ●──0     (1,0)  ●──1
   +────────────→ x
     0        1
```

In this grid:

    Moving horizontally (x-axis) changes input 1

    Moving vertically (y-axis) changes input 2

It’s standard to map:

    x₁ → x-axis

    x₂ → y-axis

This makes decision boundaries easy to interpret as straight lines (in the case of a single-layer perceptron) or curves/piecewise lines (for MLPs).

🧠 Part 2: Why an MLP Can Learn XOR (and a Perceptron Can't)
🤖 Perceptron (Single Layer)

A perceptron can only learn functions where the output classes are linearly separable — i.e., can be split by a straight line (or hyperplane in higher dimensions).

XOR is not linearly separable, as you've seen:

    There is no line that separates the (0,0) and (1,1) outputs from the (1,0) and (0,1) outputs

🔥 MLP (Multi-Layer Perceptron)

A 2-2-1 network means:

    2 input neurons

    2 hidden neurons (1 hidden layer)

    1 output neuron

Each neuron in the hidden layer computes its own weighted sum and nonlinear activation. This enables the MLP to transform the input space before making a decision.
Intuition:

    Hidden neurons can create new features or partition space nonlinearly

    The output neuron combines these nonlinear features

So instead of being limited to a single line, the network can form composite decision boundaries like this:

```
Step 1: Hidden Layer Neuron 1 draws Line A (e.g., separate (0,1) and (1,0) from others)
Step 2: Hidden Layer Neuron 2 draws Line B (e.g., separates the other region)

Output neuron then "ORs" or "XORs" the two regions.
```

These intermediate layers allow the network to solve nonlinearly separable problems by chopping the space into smaller linear regions and combining them.

✅ Summary: Can a 2-2-1 Network Learn XOR?

Yes — a single hidden layer with 2 neurons is sufficient. You do not need 2 hidden layers.

So:

    1 hidden layer with ≥ 2 neurons → ✅ XOR can be learned

    1 neuron in hidden layer → 🚫 Not enough capacity

    2 hidden layers → Overkill, but still works

🧭 Where Visualization Helps

If we plot:

    The input space as a grid (with color-coded class labels)

    The decision boundary learned by a network

We can see that:

    A single perceptron can only draw a line

    An MLP draws a piecewise-linear boundary that isolates the XOR class regions

Want me to generate a real visualization of decision boundaries for:

    Single-layer perceptron on XOR (fails)

    MLP on XOR (succeeds)

Let me know if you’d prefer a plot or want to implement and step through it yourself.

Prompt: Ok, I will want to plot the input space as a grid and the decision boundary learned by a network, but I think we need to clean up our plotting functionality. Specifically, we should start creating helper functions for plotting and creating sensible CLI args for different plotting options. Right now, we have the option to plot the learning curve and a separate plot with weights and bias curves. I think we should make it possible to pass in a list of plots that should be generated with options like `learning`, `weights` (which will include the bias), and `decision-boundary`. I will ultimately want to create an animated visualization like a gif, so we should reserve a reasonable way to provide that option as well. For reference, this is the current main.py:

<pasted in main.py>

Response: Great — we can clean this up with a modular, extensible plotting system and a smarter CLI interface. Here's the plan:

✅ Goals for Refactoring

    Modular plotting:

        Separate helper functions for each plot type: learning, weights, decision-boundary

    CLI argument for selecting multiple plots:
```sh
--plots learning weights decision-boundary
```

Future support: Reserve --animate or --gif for animated decision boundary visualization

✅ Step 1: Update CLI

Update the argparse section like this:

<provided code>

✅ Step 2: Refactor train_and_plot with Helpers

Create three new helper functions in main.py (or better, move to a plot_utils.py if you prefer later):

<provided code>

✅ Step 3: Refactor train_and_plot to Use Helpers

<provided code>

✅ Step 4: Update Calls

<provided code>

✅ Example Usages

Plot weights and learning curve:
```sh
python main.py --plots learning weights
```

Plot XOR decision boundary only:
```sh
python main.py --plots decision-boundary --skip-nand
```

Run all, no plots:
```sh
python main.py
```

Would you like me to consolidate the plotting functions into a separate module next (e.g., plot_utils.py) or jump to building the animation capability?

Prompt: Yes, let's put those in a separate module

Response: Great — let's move the plotting helpers to a clean, separate file called plot_utils.py. This will make your main.py easier to manage and set you up nicely for extending things like animations or exportable plots later.

✅ Step 1: Create plot_utils.py

Create a new file plot_utils.py alongside main.py and add this:

<provided code>

✅ Step 2: Update main.py to Use Plot Utils

At the top of main.py, import the new functions:

<provided code>

Now you can remove the old plotting code from main.py.

✅ Benefits of This Refactor

    Clear separation of concerns

    Easy to maintain and expand (e.g., add plot_animation later)

    Cleaner main.py — focused on logic, not plotting

Ready to move on to generating an animated decision boundary GIF as training progresses? We can do that by rendering and saving plots for selected epochs.

Prompt: Could you provide the full main.py? Then we'll go on to plotting decision boundaries

Response: Absolutely — here's your fully updated and cleaned-up main.py, using plot_utils.py for modular plotting.

<provided code>

✅ Next Step

Now that the code is clean and modular, we're in a great place to add animated decision boundary visualization (e.g., for XOR training).

Let me know how you'd like to handle it:

    Save the animation as a GIF or show it live?

    Plot decision boundary every N epochs?

    Animate both XOR and NAND or just one?

Ready to proceed?

Prompt: Can I view the animation live and have the option to save it as a gif, similar to how I can optionally save the plot that opens up?

Are you saying plotting every epoch will be too computationally intensive? If so, yes, we can reduce the frequency

Can we keep in place the existing options to skip either xor or nand to decide which to animate?

Response: Yes — we can absolutely do all of that, cleanly and flexibly:

✅ Goals for Animation

    Live view: Display the animation in a window using matplotlib.animation.FuncAnimation

    Optional GIF save: Use --gif to trigger saving to file

    Custom frequency: e.g., plot every 100 epochs for performance

    Respect --skip-nand and --skip-xor as you already do

    Keep code modular: we'll put the animation logic in plot_utils.py as a new function

✅ Step 1: Add animate_decision_boundary() to plot_utils.py

<provided code for animation function>

✅ Step 2: Update train_and_plot() in main.py

<just updated import in main.py>

Then inside train_and_plot(...):

```python
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
```

You’ll just need to update the function signature to:
```python
def train_and_plot(data, name: str, plots: list[str], gif: bool):
```
And call it with gif=args.gif.

✅ Step 3: Update main() to Pass args.gif

<basic function call updates>

✅ Example Usage

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

Would you like:

    An option to control --interval (frames per step)?

    To also support saving learning/weights plots to file?

Let me know how polished you want this to get!

Prompt: Would it be possible to have a gif of the decision boundary alongside gifs of the learning curve and weight and bias evolution?

Response: Yes — that's absolutely possible and actually a great idea for a complete visual record of training!

Here's how we’ll approach it:

✅ What We’ll Do

For each of the following plots:

    Decision boundary

    Learning curve

    Weight and bias evolution

We’ll generate:

    An animated version (GIF)

    Optionally, show it live (plt.show())

✅ Strategy Per Plot
1. Decision Boundary

✅ Already animated using FuncAnimation with snapshots of weights + bias
2. Learning Curve

📉 We’ll animate the error over time, building up frame-by-frame
3. Weight/Bias Evolution

📈 We’ll animate line plots of w0, w1, and bias across epochs

✅ Step-by-Step Plan

We'll create two new functions in plot_utils.py:
🔧 animate_learning_curve()

<provided code>

🔧 animate_weights_and_bias()

<provided code>

✅ Step 2: Update train_and_plot() to Call Animations

Inside train_and_plot(...) in main.py:

<provided code for main.py>

✅ Final Result

When you run:

```sh
python main.py --plots decision-boundary --gif --skip-nand
```

You’ll get:

    ✅ Live view of the final plots

    ✅ Saved GIFs:

        xor_decision_boundary.gif

        xor_learning_curve.gif

        xor_weights_bias.gif

Would you like:

    CLI options for setting the epoch interval or frame speed?

    A --save-plots flag for non-animated .png export?

Let me know how you'd like to extend this further!
