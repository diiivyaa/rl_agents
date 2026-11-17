# Value-Based Reinforcement Learning

Implementation and evaluation of **SARSA, Q-Learning, and Deep Q-Networks (DQN)** using Python and PyTorch. The project explores on-policy versus off-policy learning, exploration strategies, and deep reinforcement learning for control tasks.

## 1. Cliff Walking — SARSA vs Q-Learning

Implemented tabular SARSA and Q-Learning in a cliff-walking environment with a −300 penalty for falling off the cliff.

**Key Results:**
- With fixed ε = 0.1, SARSA learned a safer 9-step path, while Q-Learning learned the shortest 7-step path.
- During greedy evaluation, Q-Learning achieved a return of **−7**, compared to **−9** for SARSA.
- With decaying ε, SARSA also converged to the shortest 7-step path, achieving **−7** during evaluation.

These results demonstrate the trade-off between exploration safety and optimal greedy performance.

## 2. Inverted Pendulum — Deep Q-Network

Implemented a CNN-based DQN agent to control an inverted pendulum using stacked grayscale image observations.

**Architecture:**
- Input: 4 × 42 × 42 grayscale frames
- Two convolutional layers with ReLU
- Fully connected layer with 256 neurons
- Output: 9 discrete torque-action Q-values
- **88,633 trainable parameters**

The agent uses experience replay, ε-greedy exploration, and a periodically synchronized target network.

### Training

- **Training steps:** 150,000
- **GPU:** NVIDIA GeForce RTX 3070 Laptop GPU
- **Training time:** 978.27 seconds (~16.3 minutes)
- **Best checkpoint:** 115,000 environment steps

### Independent Evaluation

Evaluated the selected DQN against a random-action baseline over 20 test episodes.

| Metric | DQN | Random Baseline |
|---|---:|---:|
| Mean episode return | **−174.06** | −1207.92 |
| Standard deviation | **108.69** | 309.22 |
| Episodes evaluated | 20 | 20 |

**The DQN outperformed the random-action baseline in all 20 test episodes**, improving the mean episode return by 1033.86.

## Tech Stack

Python · PyTorch · Gymnasium · NumPy · Matplotlib

## Reference

Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013).
