# DQN-Multiagent-ATARI-Boxing
Project that teaches 2 agents box each other.

## Experimenting with Architectures:

The following comparisons were inspected: 

  1. Replay Buffer Prefilling: Prefill the replay buffer for one agent while leaving it empty for the other. Evaluate the impact on learning dynamics and overall performance.

  2. Feature Extraction Enhancement: Increase the number of features extracted using convolutions for one agent, keeping the other agent's feature extraction unchanged. Assess how this modification influences the agents' ability to learn and adapt.

  3. Stochastic Elements: Introduce noisy layers for one agent to add a level of stochasticity in its decision-making process. Simultaneously, employ an epsilon scheduler for the other agent to control exploration-exploitation trade-offs.


| Comparison               | Final learned models               |  Learning plots              |
| ---------------------- | ---------------------- | ---------------------- |
| eps vs noisy                | ![v1](results/gifs/eps-vs-noisy.gif) | <img src="results/figures/2-xtra-small_1600-init_noisy-eps-1.png" width="1000" height="220">| 
| refill                | ![v1](results/gifs/refill_vs_emptyBuffer.gif) | <img src="results/figures/2-xtra-small_5000A1-0A2_2x-noisy-1.png" width="1000" height="220"> |
| architectures                | ![v1](results/gifs/small_vs_xtra-small.gif) | <img src="results/figures/xtra-small-small_1600-init_2x-noisy-1.png" width="1000" height="220"> |


## Installation
```console
pip install -r requirements.txt
```
Installation of ROMs: https://pettingzoo.farama.org/environments/atari/#installation

Note: You might have to downgrade Python to 3.8.18.
