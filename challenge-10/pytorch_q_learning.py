import torch
import matplotlib.pyplot as plt

# Use Metal backend if on Apple Silicon
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

BOARD_ROWS, BOARD_COLS, NUM_ACTIONS = 5, 5, 4
START = torch.tensor([0, 0], dtype=torch.long, device=device)
WIN_STATE = torch.tensor([4, 4], dtype=torch.long, device=device)
HOLE_STATE = torch.tensor(
    [[1, 0], [3, 1], [4, 2], [1, 3]], dtype=torch.long, device=device
)

Q = torch.zeros(
    (BOARD_ROWS, BOARD_COLS, NUM_ACTIONS), dtype=torch.float32, device=device
)

try:
    profile
except NameError:

    def profile(func):
        return func


def get_reward(states):
    rewards = torch.full((states.shape[0],), -1.0, device=device)
    is_hole = (states[:, None] == HOLE_STATE).all(dim=2).any(dim=1)
    is_win = (states == WIN_STATE).all(dim=1)
    rewards[is_hole] = -5.0
    rewards[is_win] = 1.0
    return rewards


def compute_next_states(states, actions):
    deltas = torch.zeros_like(states, device=device)
    deltas[:, 0] = (actions == 1).long() - (actions == 0).long()
    deltas[:, 1] = (actions == 3).long() - (actions == 2).long()
    next_states = states + deltas
    return torch.clamp(next_states, 0, 4)


def update_q(Q, states, actions, rewards, next_states, alpha=0.5, gamma=0.9):
    indices = states[:, 0] * BOARD_COLS + states[:, 1]
    next_indices = next_states[:, 0] * BOARD_COLS + next_states[:, 1]

    q_current = Q.view(-1, NUM_ACTIONS)[indices, actions]
    q_next = Q.view(-1, NUM_ACTIONS)[next_indices]
    q_next_max = q_next.max(dim=1).values
    q_target = rewards + gamma * q_next_max

    q_updated = (1 - alpha) * q_current + alpha * q_target
    Q.view(-1, NUM_ACTIONS)[indices, actions] = q_updated


@profile
def run_pytorch_q_learning(num_agents=512, max_episodes=1000, max_steps_per_episode=50):
    global Q
    Q.zero_()
    rewards_history = []

    for _ in range(max_episodes):
        states = START.repeat(num_agents, 1)
        done = torch.zeros(num_agents, dtype=torch.bool, device=device)
        episode_rewards = torch.zeros(num_agents, dtype=torch.float32, device=device)

        steps = 0
        while not done.all() and steps < max_steps_per_episode:
            actions = torch.randint(0, NUM_ACTIONS, (num_agents,), device=device)
            next_states = compute_next_states(states, actions)
            rewards = get_reward(next_states)
            update_q(Q, states, actions, rewards, next_states)

            episode_rewards += rewards
            done |= (rewards == 1.0) | (rewards == -5.0)
            states = torch.where(done.unsqueeze(1), START, next_states)
            steps += 1

        rewards_history.append(episode_rewards.sum().item())

    return rewards_history


if __name__ == "__main__":
    # Confirm that Apple silicon metal performance shaders are available
    # print(torch.backends.mps.is_available())
    rewards = run_pytorch_q_learning(max_episodes=10000)
    plt.plot(rewards)
    plt.title("Total Rewards (PyTorch)")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.grid(True)
    plt.show()
