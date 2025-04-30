import torch
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Board and environment settings
BOARD_ROWS, BOARD_COLS, NUM_ACTIONS = 5, 5, 4
START = torch.tensor([0, 0], dtype=torch.long, device=device)
WIN_STATE = torch.tensor([4, 4], dtype=torch.long, device=device)
HOLE_STATE = torch.tensor(
    [[1, 0], [3, 1], [4, 2], [1, 3]], dtype=torch.long, device=device
)

# Initialize Q-table
Q = torch.zeros(
    (BOARD_ROWS, BOARD_COLS, NUM_ACTIONS), dtype=torch.float32, device=device
)

# Create hole mask
HOLE_MASK = torch.zeros((BOARD_ROWS, BOARD_COLS), dtype=torch.bool, device=device)
HOLE_MASK[HOLE_STATE[:, 0], HOLE_STATE[:, 1]] = True


def get_reward(states):
    rewards = torch.full((states.shape[0],), -1.0, device=device)
    hole_mask = HOLE_MASK[states[:, 0], states[:, 1]]
    win_mask = (states[:, 0] == WIN_STATE[0]) & (states[:, 1] == WIN_STATE[1])
    rewards[hole_mask] = -5.0
    rewards[win_mask] = 1.0
    return rewards


def compute_next_states(states, actions):
    next_states = states.clone()
    next_states[actions == 0, 0] -= 1  # up
    next_states[actions == 1, 0] += 1  # down
    next_states[actions == 2, 1] -= 1  # left
    next_states[actions == 3, 1] += 1  # right
    next_states = torch.clamp(next_states, 0, 4)
    return next_states


def update_q(Q, states, actions, rewards, next_states, alpha=0.5, gamma=0.9):
    q_current = Q[states[:, 0], states[:, 1], actions]
    q_next_max = Q[next_states[:, 0], next_states[:, 1]].max(dim=1).values
    q_target = rewards + gamma * q_next_max
    Q[states[:, 0], states[:, 1], actions] = (1 - alpha) * q_current + alpha * q_target


# Simulation settings
NUM_AGENTS = 512
MAX_STEPS = 10000
states = START.repeat(NUM_AGENTS, 1)
rewards_history = []

for step in range(MAX_STEPS):
    actions = torch.randint(0, NUM_ACTIONS, (NUM_AGENTS,), device=device)
    next_states = compute_next_states(states, actions)
    rewards = get_reward(next_states)
    update_q(Q, states, actions, rewards, next_states)
    rewards_history.append(rewards.sum().item())
    done = (rewards == 1.0) | (rewards == -5.0)
    states = torch.where(done.unsqueeze(1), START, next_states)

# Plot
plt.plot(rewards_history)
plt.title("Total Rewards over Time (PyTorch)")
plt.xlabel("Episode")
plt.ylabel("Sum of Rewards")
plt.grid(True)
plt.show()
