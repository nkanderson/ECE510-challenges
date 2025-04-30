import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_ROWS, BOARD_COLS, NUM_ACTIONS = 5, 5, 4
START = torch.tensor([0, 0], dtype=torch.long, device=device)
WIN_STATE = torch.tensor([4, 4], dtype=torch.long, device=device)
HOLE_STATE = torch.tensor(
    [[1, 0], [3, 1], [4, 2], [1, 3]], dtype=torch.long, device=device
)

Q = torch.zeros(
    (BOARD_ROWS, BOARD_COLS, NUM_ACTIONS), dtype=torch.float32, device=device
)

HOLE_MASK = torch.zeros((BOARD_ROWS, BOARD_COLS), dtype=torch.bool, device=device)
HOLE_MASK[HOLE_STATE[:, 0], HOLE_STATE[:, 1]] = True

try:
    profile
except NameError:

    def profile(func):
        return func


def get_reward(states):
    rewards = torch.full((states.shape[0],), -1.0, device=device)
    hole_mask = HOLE_MASK[states[:, 0], states[:, 1]]
    win_mask = (states[:, 0] == WIN_STATE[0]) & (states[:, 1] == WIN_STATE[1])
    rewards[hole_mask] = -5.0
    rewards[win_mask] = 1.0
    return rewards


def compute_next_states(states, actions):
    next_states = states.clone()
    next_states[actions == 0, 0] -= 1
    next_states[actions == 1, 0] += 1
    next_states[actions == 2, 1] -= 1
    next_states[actions == 3, 1] += 1
    return torch.clamp(next_states, 0, 4)


def update_q(Q, states, actions, rewards, next_states, alpha=0.5, gamma=0.9):
    q_current = Q[states[:, 0], states[:, 1], actions]
    q_next_max = Q[next_states[:, 0], next_states[:, 1]].max(dim=1).values
    q_target = rewards + gamma * q_next_max
    Q[states[:, 0], states[:, 1], actions] = (1 - alpha) * q_current + alpha * q_target


@profile
def run_pytorch_q_learning(num_agents=512, max_steps=10000):
    global Q  # Reuse Q-table across runs
    states = START.repeat(num_agents, 1)
    rewards_history = []
    for step in range(max_steps):
        actions = torch.randint(0, NUM_ACTIONS, (num_agents,), device=device)
        next_states = compute_next_states(states, actions)
        rewards = get_reward(next_states)
        update_q(Q, states, actions, rewards, next_states)
        rewards_history.append(rewards.sum().item())
        done = (rewards == 1.0) | (rewards == -5.0)
        states = torch.where(done.unsqueeze(1), START, next_states)
    return rewards_history


if __name__ == "__main__":
    rewards = run_pytorch_q_learning()
    # plt.plot(rewards)
    # plt.title("Total Rewards (PyTorch)")
    # plt.xlabel("Episode")
    # plt.ylabel("Sum of Rewards")
    # plt.grid(True)
    # plt.show()
