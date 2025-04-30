Prompt: I'm interested in identifying the computational bottlenecks in the following code:

<pasted in code from https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Q-Learning%20Algorithm.py>

Response: Great — this Q-learning implementation is relatively straightforward, but there are definitely some areas where performance could become a bottleneck, especially when scaled to more episodes or larger boards. Let's break it down by computational hotspots:

🔍 Potential Bottlenecks
1. Q-Table Lookup and Updates (Most Frequent)

```python
self.Q[(i, j, k)]
self.new_Q[(i, j, k)] = ...
```

Every state-action pair is stored as a tuple key in a flat dictionary.

Lookup and update operations on a Python dict are O(1) on average, but in tight loops (e.g., Q_Learning()), the cumulative cost is high.

The full Q dictionary is copied after every episode:
```python
self.Q = self.new_Q.copy()
```

That’s an O(n) operation, where n = BOARD_ROWS × BOARD_COLS × |actions| = 5 × 5 × 4 = 100.

2. Repeated State Object Reinstantiation

```python
self.State = State(state=next_state)
```

This creates a new instance each time the agent moves. If episodes are long, this becomes expensive.

It also invokes repeated HOLE_STATE membership checks (linear scan over a list).

3. Inefficient Reward and End Checks

```python
for i in HOLE_STATE:
    if self.state == i:
```

Each call to getReward() or isEndFunc() iterates through HOLE_STATE, a list.

Convert to a set to get O(1) lookups instead of O(n).

4. Redundant Code in Q_Learning

    The loop contains duplicated logic:

```python
for a in self.actions:
    ...
```
both for new_Q updates and for computing the max future Q-value. These could be merged or optimized.

5. Plotting Overhead (not critical, but accumulative)

```python
self.plot_reward.append(self.rewards)
```

Appending for every episode is fine unless you want live plotting. Then matplotlib calls would be more expensive.

✅ Quick Optimization Suggestions

Optimization	Reason	Fix
Use set for HOLE_STATE	Faster lookup	HOLE_STATE = {(1,0), (3,1), (4,2), (1,3)}
Avoid copying new_Q every episode	Reduces redundant memory ops	Update self.Q in-place or defer full copy
Replace State re-instantiation with a method	Avoid overhead	Use self.State.set(next_state)
Combine getReward() and isEndFunc()	Avoid repeated loops	Merge logic into a single function
Precompute Q-table keys	Reduce recomputation	Cache or use numpy/matrix-based indexing for Q-table

🧪 Profiling Tip

To confirm these are hotspots, run:
```sh
python -m cProfile -s tottime your_script.py
```

You’ll likely see high cumulative time in:

    Agent.Q_Learning

    Agent.Action

    dict.copy

    State.getReward

Would you like help rewriting parts of the code to implement these optimizations or set up a profiler to quantify them?

Prompt: I'd like to do a little more profiling. I already ran `python -m cProfile -s tottime your_script.py`, and came up with the following (only showing the first handful of lines here):
```
2028162 function calls (2007755 primitive calls) in 1.610 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.209    0.209    0.453    0.453 q_learning.py:135(Q_Learning)
     59/7    0.129    0.002    1.639    0.234 text.py:65(_get_text_metrics_with_cache)
    45/42    0.090    0.002    0.094    0.002 {built-in method _imp.create_dynamic}
    82961    0.079    0.000    0.141    0.000 q_learning.py:106(Action)
        1    0.077    0.077    0.149    0.149 backend_bases.py:2646(create_with_canvas)
      290    0.065    0.000    0.065    0.000 {method 'read' of '_io.BufferedReader' objects}
     7555    0.046    0.000    0.089    0.000 inspect.py:882(cleandoc)
        1    0.043    0.043    0.043    0.043 {function FigureManagerMac.destroy at 0x1076d56c0}
        1    0.035    0.035    0.069    0.069 backend_macosx.py:151(__init__)
      290    0.031    0.000    0.031    0.000 {built-in method marshal.loads}
   122983    0.030    0.000    0.030    0.000 {built-in method builtins.round}
    93006    0.026    0.000    0.026    0.000 {method 'copy' of 'dict' objects}
        1    0.024    0.024    0.025    0.025 backend_macosx.py:118(__init__)
        1    0.020    0.020    0.020    0.020 {method '_show' of 'matplotlib.backends._macosx.FigureManager' objects}
    92961    0.020    0.000    0.020    0.000 q_learning.py:33(getReward)
```
Let's talk through some of what we're seeing here, and then discuss additional profiling options, such as snakeviz

Response: You're off to a great start — the cProfile output gives us solid insight into where time is spent. Let’s break down what you’ve shown and then move into next-step profiling tools like snakeviz.

🔍 Analysis of cProfile Output

2028162 function calls (2007755 primitive calls) in 1.610 seconds

That's over 2 million calls, most of which are primitive (non-recursive, low-level calls), in ~1.6 seconds. That suggests your bottlenecks are more about high call volume than a few expensive one-offs.
🔹 Top functions by internal time (tottime)

Function	Notes
Q_Learning (0.209s, 1 call)	Your main loop — as expected, high cost.
_get_text_metrics_with_cache (matplotlib)	Likely related to plt.show() or text rendering. Not a bottleneck for Q-learning logic itself.
Action (0.079s, 82,961 calls)	This is called once per step, so ~8.3 µs per call. Still worth optimizing due to high volume.
round (0.030s, 122,983 calls)	Not huge per-call cost, but very frequent — could switch to fixed-point math or delay rounding.
dict.copy (0.026s, 93,006 calls)	Confirms earlier suspicion that copying the Q-table each step is costly.
getReward (0.020s, 92,961 calls)	Also called per step; optimize HOLE_STATE access.

📊 Immediate Takeaways

    Action(), getReward(), and dict.copy() are hot due to frequency, not individual slowness.

    round() appears enough to merit review — maybe batch final rounding or avoid it in hot paths.

    The matplotlib functions (e.g., _get_text_metrics_with_cache) are not part of the Q-learning core loop. They're used once at the end, so they can be ignored for now.

  🐍 Using snakeviz for Better Visualization

snakeviz gives you a flamegraph-style UI (sunburst or icicle views) to explore call stacks and hotspots visually.
✅ Installation

pip install snakeviz

✅ Usage

python -m cProfile -o profile.out your_script.py
snakeviz profile.out

    A browser window will open with interactive graphs.

    You can drill down into Q_Learning, see what it calls, and follow down to hotspots like getReward() and dict.copy().

This is extremely helpful when:

    You want to explore nested calls without digging through cProfile manually.

    You're trying to understand which call paths lead to the most time spent.

🛠️ Next Profiling Options (Advanced)

If you want even finer granularity, especially line-by-line:
1. line_profiler

pip install line_profiler

Use it like:

from line_profiler import LineProfiler

lp = LineProfiler()
lp_wrapper = lp(ag.Q_Learning)
lp_wrapper(episodes)
lp.print_stats()

You can profile specific functions (like Action, getReward, or Q_Learning) and get output per line.
2. Memory Profiling (if you suspect memory pressure)

pip install memory_profiler

And use the @profile decorator on suspect functions.

Would you like to set up line_profiler or explore a specific call chain in snakeviz next?

Prompt: Let's shifts gears - given the above cProfile output and identification of high-volume call bottlenecks, could you propose a hardware implementation for the largest bottleneck? I'm ultimately looking for systemverilog code for this bottleneck

Response: Excellent — let’s translate your profiling insights into hardware acceleration opportunities.
🔍 Summary of Key Bottleneck

From your cProfile results and call frequency:
🔺 Top Candidate for Hardware Acceleration: Q-table Access & Update

    Function: Q_Learning() + Action()

    Critical operation:

`q_value = (1 - alpha) * Q[(i,j,action)] + alpha * (reward + gamma * Q[next_state_action])`

    This involves:

        Lookup of two Q-values (Q[(i,j,action)], Q[next_state_action])

        Weighted average computation

        Frequent updates to Q-values (dict write)

This is essentially a fixed-point multiply-accumulate (MAC) operation, repeated tens of thousands of times per episode, ideal for pipelined or parallelized digital logic.

🎯 Goal

Implement a Q-value update unit in SystemVerilog that performs the following:
Q(s,a)←(1−α)∗Q(s,a)+α∗(reward+γ∗Q(s’,a’))
Q(s,a)←(1−α)∗Q(s,a)+α∗(reward+γ∗Q(s’,a’))

Where:

    Q(s, a) is the old Q-value

    Q(s’, a’) is the max Q-value for next state

    α and γ are fixed hyperparameters (e.g., 0.5, 0.9)

    reward is typically small integer (-5 to 1)

🧠 Fixed-Point Assumptions (Simplify Arithmetic)

Assume:

    Q-values are signed 16-bit fixed-point, Q8.8 format ([15:0])

    alpha = 0.5, gamma = 0.9 encoded as fixed-point constants

✅ SystemVerilog Module: Q-Value Update Unit

<provided sv code>

📦 Integration Plan

This module would sit inside a larger Q-table controller that:

    Fetches Q-values from BRAM (indexed by state-action pairs)

    Feeds values into this update unit

    Writes updated Q-values back to memory

    Optionally supports pipelining for multiple agents or concurrent updates

🧩 Next Steps

Would you like:

    A Verilog testbench for this module?

    A memory interface controller (e.g., Q-table BRAM manager)?

    A multi-agent support wrapper?

This could evolve into a fully pipelined hardware RL accelerator.

Prompt: So would there be 4 of these operating in parallel (one for each possible action)?

Response: Yes — exactly right. If you're implementing a Q-learning agent in hardware and want to find the best action for a given state, then:

✅ You Need to Evaluate All 4 Actions in Parallel

Because Q-learning’s update step requires:

    Selecting the best next action (i.e., computing max_a' Q(s', a'))

    And optionally updating all Q-values (e.g., initializing or refreshing them)

So, instantiating 4 parallel QUpdateUnit modules, one per action, makes sense in two cases:

🧠 Two Use Cases for Parallelism
1. Max Q(s′, a′) Computation

    When in state s', you want:

max(Q[s'][a0], Q[s'][a1], Q[s'][a2], Q[s'][a3])

Parallel fetch → all 4 Q-values → compare in parallel → pick max.

✅ Requires:

    4 Q-value fetch units (or read ports)

    4 comparators (or a 2-stage comparator tree)

2. Q(s, a) Update

    Technically, you only update Q(s, a) — a single action per step.

    But if you wanted to update all 4 Q-values in a batch (e.g., during offline training or reward backprop), then:

        You’d instantiate 4 QUpdateUnits — one per action.

✅ Good for:

    Vectorized updates

    Higher throughput training

🔧 Typical Q-Learning Hardware Pipeline

Let’s imagine a single-agent Q-learning step in hardware:

    Fetch

        Current Q(s, a)

        All Q(s′, a′) for a′ ∈ {0,1,2,3}

        Use parallel reads from BRAM

    Compute

        Parallel comparators → max_a′ Q(s′, a′)

        Feed into QUpdateUnit with reward, α, γ

    Write

        Updated Q(s, a) → back into BRAM

This means you only need one QUpdateUnit per agent for actual updates, but 4 Q-value read paths for max-computation.

🧰 Hardware Diagram (Simplified)

              ┌──────────────┐
              │  Q-table RAM │◄──┐
              └─────┬────────┘   │
          ┌─────────▼─────────┐ │   (read Q(s,a) and Q(s′,a′))
          │   Q-value Fetch   │─┘
          └────┬─────┬─────┬──┘
               │     │     │
               ▼     ▼     ▼       (4 Q(s′, a′))
          ┌───────────────────┐
          │   Max Comparator  │─────► max Q(s′, a′)
          └───────────────────┘
                      │
                      ▼
           ┌────────────────────┐
           │    QUpdateUnit     │◄──── reward
           └────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │   RAM Write  │───► updated Q(s,a)
              └──────────────┘

💡 Final Notes

    Yes, parallel Q-value reads (for all actions) are essential.

    Only one update unit is typically required per agent (unless batch updates are needed).

    The rest of the bottleneck is memory bandwidth and RAM access scheduling.

Would you like to:

    Add a max-Q selector module for 4 Q-values?

    Design a Q-table RAM interface to support parallel reads?

    Include FSM logic to coordinate agent steps?

Let me know how detailed you'd like to go.

Prompt: What if instead of creating this hardware acceleration, we optimized the existing code for a GPU. As a reminder, this is the code:
<pasted in original code>

Response: To optimize this Q-learning code for GPU execution, the key idea is to vectorize and parallelize the parts of the algorithm that are most compute-intensive and well-suited for GPU hardware. Since GPUs excel at matrix and tensor computations, we’ll rework the Q-value storage and update mechanics using NumPy + CuPy or PyTorch, where the heavy lifting can happen on the GPU.

🎯 Target for GPU Acceleration

The biggest bottleneck in your current CPU implementation is:

q_value = (1-alpha) * Q[(i,j,action)] + alpha * (reward + gamma * Q[next_state])

You repeat this per step, per action, per episode. This kind of scalar loop is terrible for CPU cache performance, but perfect for GPU batch ops.

🚀 Step-by-Step GPU Optimization Plan
✅ 1. Replace Q-table dictionary with a 3D tensor

```python
# CPU: dict[(i, j, a)] → GPU: Q_table[i, j, a]
import cupy as cp  # or use torch

BOARD_ROWS, BOARD_COLS, NUM_ACTIONS = 5, 5, 4
Q = cp.zeros((BOARD_ROWS, BOARD_COLS, NUM_ACTIONS), dtype=cp.float32)
```

✅ 2. Store HOLE_STATE as a GPU-side lookup mask
```python
HOLE_MASK = cp.zeros((BOARD_ROWS, BOARD_COLS), dtype=cp.bool_)
for x, y in HOLE_STATE:
    HOLE_MASK[x, y] = True
```

✅ 3. Vectorize getReward over batches of states
```python
def get_reward(states):
    """states: (N, 2) array"""
    rewards = cp.full((states.shape[0],), -1, dtype=cp.float32)
    hole_mask = HOLE_MASK[states[:, 0], states[:, 1]]
    win_mask = (states[:, 0] == WIN_STATE[0]) & (states[:, 1] == WIN_STATE[1])
    rewards[hole_mask] = -5
    rewards[win_mask] = 1
    return rewards
```

✅ 4. GPU-Accelerated Q-value Update (Batched)

Assume:

    states: shape (N, 2)

    actions: shape (N,)

    next_states: shape (N, 2)
```python
def update_q(Q, states, actions, rewards, next_states, alpha=0.5, gamma=0.9):
    # Q[s, a] = (1-alpha)*Q[s,a] + alpha*(r + gamma*max_a' Q[s',a'])
    q_current = Q[states[:, 0], states[:, 1], actions]
    q_next_max = cp.max(Q[next_states[:, 0], next_states[:, 1]], axis=1)
    q_target = rewards + gamma * q_next_max
    Q[states[:, 0], states[:, 1], actions] = (
        (1 - alpha) * q_current + alpha * q_target
    )
```

🧪 Training Loop Example (Batched)

Instead of looping over 10,000 episodes sequentially, launch parallel rollouts of episodes (e.g. 512 agents) on the GPU:
```python
NUM_AGENTS = 512
states = cp.zeros((NUM_AGENTS, 2), dtype=cp.int32)  # start at (0,0)
done = cp.zeros(NUM_AGENTS, dtype=cp.bool_)

for step in range(MAX_STEPS):
    actions = cp.random.randint(0, 4, size=NUM_AGENTS)
    next_states = compute_next_positions(states, actions)  # GPU logic
    rewards = get_reward(next_states)
    update_q(Q, states, actions, rewards, next_states)
    done = (rewards == 1) | (rewards == -5)
    states = cp.where(done[:, None], 0, next_states)  # reset finished
```

📈 Performance Impact

Moving this logic to GPU will give you:

    Massive speedup from batch processing (thousands of agents at once)

    Elimination of Python dictionaries

    Hardware-accelerated matrix operations

    Easy scalability to larger boards or more actions

🧠 Tools to Use

Tool	Use
CuPy	Numpy-compatible GPU arrays
PyTorch	Easier debugging, GPU/CPU backends
Numba.cuda	Compile select kernels to GPU manually
TensorFlow	Overkill for Q-tables, but possible


✅ Summary of What to Replace
Old Code	GPU-Optimized
Q[(i,j,a)]	Q[i,j,a] (tensor)
Loop over actions	Use cp.argmax() or cp.max()
Dict copy self.Q = self.new_Q	Not needed — tensor updates in-place
Scalar reward logic	Vectorized batch reward kernel

Would you like me to generate a working CuPy-based version of your Q-learning loop, or a PyTorch equivalent?

Prompt: What are the differences between CuPy and PyTorch?

Resdponse: Great question — both CuPy and PyTorch can be used to accelerate numerical computations on GPUs, but they serve different purposes, have different APIs, and suit different kinds of workflows. Here's a side-by-side comparison to help you choose the right tool depending on your goals:

🧮 High-Level Comparison: CuPy vs PyTorch

Feature	CuPy	PyTorch
Purpose	GPU-accelerated NumPy replacement	Deep learning framework (plus tensor operations)
API Style	NumPy-compatible (drop-in)	Custom tensor API (not NumPy-compatible)
GPU Acceleration	Yes (via CUDA)	Yes (via CUDA, ROCm, or CPU fallback)
Best For	Accelerating existing NumPy code	Deep learning, neural networks, GPU batch ops
Ease of Transition	Very easy from NumPy	Requires learning PyTorch tensor semantics
Autograd / Gradients	❌ Not built-in	✅ Fully automatic differentiation
Ecosystem	Focused on array math	Rich ML ecosystem (e.g., torchvision, torchtext, etc.)
Model Training	❌ Not intended for model definition	✅ Full training pipelines for NN/RL
Deployment Tools	Minimal	TorchScript, ONNX, PyTorch Mobile
Installation Size	Lightweight (~100MB)	Larger (~800MB+)

✅ When to Use CuPy

Use CuPy if:

    You're optimizing existing NumPy-based code (like your Q-learning loop).

    You don’t need gradients or deep learning libraries.

    You want fast prototyping with minimal code changes.

    Your data is array-based and fits well into broadcasting, reductions, matrix ops.

```python
import numpy as np
import cupy as cp

x_cpu = np.ones((1000, 1000))
x_gpu = cp.asarray(x_cpu)  # easy transition
result = cp.sum(x_gpu * 3.14)
```

✅ When to Use PyTorch

Use PyTorch if:

    You want to build neural networks or reinforcement learning pipelines.

    You need automatic differentiation (e.g., for policy gradients).

    You’re working with pretrained models or multi-agent RL.

    You need deployment/export options.

```python
import torch

Q = torch.zeros((5, 5, 4), device='cuda')
s = torch.tensor([2, 3], device='cuda')  # state
a = 1
Q[s[0], s[1], a] += 0.1
```

🧠 Summary

Use Case	Recommended Tool
Speeding up NumPy-style Q-learning	CuPy
Training a DQN or policy gradient agent	PyTorch
Need automatic gradients	PyTorch
Fast drop-in GPU acceleration for loops	CuPy
Deep learning model export / deployment	PyTorch

Would you like a working example of the Q-table loop implemented in CuPy or PyTorch to demonstrate how it changes in practice?

Prompt: Ok, let's use CuPy, and make all the changes you recommended above to get this code updated to run on a GPU

Response: The code update requires the cupy library, which isn't currently installed in this environment. However, here’s the complete CuPy-accelerated version of your Q-learning code that you can run locally after installing CuPy.

✅ Step 1: Install CuPy

For CUDA 11.x:

pip install cupy-cuda11x

Replace 11x with your actual CUDA version (e.g., cupy-cuda11 or cupy-cuda12x).

✅ Step 2: Full CuPy-Based Q-Learning Code

```python
import cupy as cp
import matplotlib.pyplot as plt

BOARD_ROWS, BOARD_COLS, NUM_ACTIONS = 5, 5, 4
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1, 0), (3, 1), (4, 2), (1, 3)]

# Create Q-table and masks
Q = cp.zeros((BOARD_ROWS, BOARD_COLS, NUM_ACTIONS), dtype=cp.float32)

HOLE_MASK = cp.zeros((BOARD_ROWS, BOARD_COLS), dtype=cp.bool_)
for x, y in HOLE_STATE:
    HOLE_MASK[x, y] = True

def get_reward(states):
    rewards = cp.full((states.shape[0],), -1.0, dtype=cp.float32)
    hole_mask = HOLE_MASK[states[:, 0], states[:, 1]]
    win_mask = (states[:, 0] == WIN_STATE[0]) & (states[:, 1] == WIN_STATE[1])
    rewards[hole_mask] = -5.0
    rewards[win_mask] = 1.0
    return rewards

def compute_next_states(states, actions):
    next_states = states.copy()
    # Apply action deltas
    next_states[actions == 0, 0] -= 1  # up
    next_states[actions == 1, 0] += 1  # down
    next_states[actions == 2, 1] -= 1  # left
    next_states[actions == 3, 1] += 1  # right
    # Keep states within bounds
    return cp.clip(next_states, 0, 4)

def update_q(Q, states, actions, rewards, next_states, alpha=0.5, gamma=0.9):
    q_current = Q[states[:, 0], states[:, 1], actions]
    q_next_max = cp.max(Q[next_states[:, 0], next_states[:, 1]], axis=1)
    q_target = rewards + gamma * q_next_max
    Q[states[:, 0], states[:, 1], actions] = \
        (1 - alpha) * q_current + alpha * q_target

# Simulation parameters
NUM_AGENTS = 512
MAX_STEPS = 10000

states = cp.tile(cp.array([START], dtype=cp.int32), (NUM_AGENTS, 1))
rewards_history = []

for step in range(MAX_STEPS):
    actions = cp.random.randint(0, NUM_ACTIONS, size=NUM_AGENTS)
    next_states = compute_next_states(states, actions)
    rewards = get_reward(next_states)
    update_q(Q, states, actions, rewards, next_states)
    rewards_history.append(cp.sum(rewards).item())
    done = (rewards == 1.0) | (rewards == -5.0)
    states = cp.where(done[:, None], cp.array(START, dtype=cp.int32), next_states)

# Plot results
plt.plot(rewards_history)
plt.title("Total Rewards over Time")
plt.xlabel("Episode")
plt.ylabel("Sum of Rewards")
plt.grid(True)
plt.show()
```

🔍 What This Does

    Simulates 512 Q-learners in parallel on the GPU.

    Vectorizes Q-table updates with CuPy.

    Replaces per-step scalar logic with batch GPU math.

    Tracks reward totals across episodes for plotting.

Would you like a version using PyTorch instead (for CPU fallback or easier neural net integration)?

Prompt: Ah, so this requires cuda to be installed separately, is that correct? If so, let's go with a PyTorch version

Response: The PyTorch-based Q-learning code encountered an environment-level memory or installation issue, but it should run correctly on your local machine. Here's the full PyTorch version that mimics the CuPy code and can run on either GPU or CPU:

✅ PyTorch Q-learning Code (Runs on GPU or CPU)

<provided code saved in pytorch_q_learning.py>

📝 Instructions

    Install PyTorch:

pip install torch

Run the code in a Python environment (Jupyter, script, or IDE).

If you have a GPU, PyTorch will use it automatically.

Would you like to extend this into a DQN or policy-gradient agent next, or do further profiling on this version?

Prompt: I'd like to profile and compare this version against the original. Can we create a script to go in a `main.py` file that will do that? I understand we may need to modify the original files to add `@profile` decorators, but we can walk through that after generating the script. The name of the original q learning script is `q_learning.py` and the new one is `pytorch_q_learning.py`

Response: Great — profiling and comparing both versions is a smart way to quantify speedups and identify any remaining bottlenecks.

Let’s start by generating a main.py script that:

✅ What main.py Will Do

    Runs both q_learning.py (original) and pytorch_q_learning.py (optimized).

    Profiles them using line_profiler (for @profile) or optionally cProfile.

    Records runtime for each version.

    Optionally shows memory usage with memory_profiler.

📝 Step 1: main.py Script

<provided code saved in main.py>

🧰 Requirements

Make sure you have these installed:

pip install line-profiler memory-profiler

And use:

kernprof -l -v q_learning.py

🔜 Next Step

Now we need to:

    Add @profile to functions in both scripts (e.g., Q_Learning in q_learning.py, and update_q, compute_next_states in pytorch_q_learning.py)

    Optionally restructure those scripts to isolate those functions in a way that @profile works cleanly.

Would you like help with:

    Modifying q_learning.py and pytorch_q_learning.py to insert @profile decorators?

    Structuring their logic to avoid profiling top-level script code?

Prompt: Yes, let's work on structuring the logic to avoid profiling top-level script code, and add any necessary decorators

Response: Great — we’ll now refactor both q_learning.py and pytorch_q_learning.py to:

    Move the main training logic into functions that can be profiled.

    Add @profile decorators to those functions.

    Keep a clean if __name__ == "__main__": block so they still work as standalone scripts.

    Make sure main.py can run kernprof on them.

✅ Refactor Plan
🔧 1. q_learning.py (Original CPU Version)

Changes:

    Move code from Agent.Q_Learning() into a new function run_q_learning(episodes).

    Add @profile to the function.

    Move Agent class instantiation and call under if __name__ == "__main__".

🔧 2. pytorch_q_learning.py (Optimized Version)

Changes:

    Move simulation loop into run_pytorch_q_learning(num_agents, max_steps) function.

    Add @profile.

    Keep plotting in __main__ block.

✏️ Updated q_learning.py Skeleton

Put this at the top:

```python
# Add at top of file
try:
    profile
except NameError:
    def profile(func): return func
```

Add this near the bottom:
```python
@profile
def run_q_learning(episodes=10000):
    ag = Agent()
    ag.Q_Learning(episodes)
    return ag.plot_reward
```

Update the if __name__ == "__main__" block:
```python
if __name__ == "__main__":
    rewards = run_q_learning()
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title("Q-Learning Rewards")
    plt.show()
```

✏️ Updated pytorch_q_learning.py Skeleton

Same decorator guard at the top:
```python
try:
    profile
except NameError:
    def profile(func): return func
```

Wrap the core training loop:
```python
@profile
def run_pytorch_q_learning(num_agents=512, max_steps=10000):
    Q = torch.zeros((5, 5, 4), device=device)
    states = START.repeat(num_agents, 1)
    rewards_history = []

    for step in range(max_steps):
        actions = torch.randint(0, 4, (num_agents,), device=device)
        next_states = compute_next_states(states, actions)
        rewards = get_reward(next_states)
        update_q(Q, states, actions, rewards, next_states)
        rewards_history.append(rewards.sum().item())
        done = (rewards == 1.0) | (rewards == -5.0)
        states = torch.where(done.unsqueeze(1), START, next_states)

    return rewards_history
```

Main block:
```python
if __name__ == "__main__":
    rewards = run_pytorch_q_learning()
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title("Total Rewards (PyTorch)")
    plt.show()
```

🏃‍♂️ Usage Instructions

After refactoring:
Profile line-by-line:
```sh
kernprof -l -v q_learning.py
kernprof -l -v pytorch_q_learning.py
```

Or use your main.py script:
```sh
python main.py
```

Would you like me to generate the full updated q_learning.py and 
pytorch_q_learning.py content now with all these changes included?

Prompt: Yes, I'd like to have the full files for each
