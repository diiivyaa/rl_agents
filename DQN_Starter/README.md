# Image-Based Pendulum Control: Student Starter

This runnable starter accompanies Question 3 of Assignment 5. Complete all six
`TODO-Q3-*` items before interpreting training results as a DQN solution.

**The initial program is a smoke-test scaffold, not a functioning DQN agent.**
It uses a linear-only model, random actions, deterministic replay selection,
a zero-gradient loss, and no target synchronization. It should run and produce
files, but it is not expected to learn pendulum control.

## Files and Tasks

| Marker | Location | Work required |
| --- | --- | --- |
| TODO-Q3-1 | dqn.py: QNetwork.__init__ | Design the CNN and linear layers |
| TODO-Q3-2 | dqn.py: QNetwork.forward | Connect the forward pass with correct dimensions |
| TODO-Q3-3 | dqn.py: DQN.choose_action | Implement epsilon-greedy actions |
| TODO-Q3-4 | replay_memory.py: ReplayBuffer.sample | Implement uniform random minibatch sampling |
| TODO-Q3-5 | dqn.py: DQN.learn | Implement selected Q-values, Bellman targets, and mean squared TD-error loss |
| TODO-Q3-6 | dqn.py: DQN.sync_target | Synchronize the Target Network |

`environment.py`, `train.py`, `run_model.py`, and `result_show.py` supply the
environment and experiment infrastructure. Preserve their observation/action
contracts. Only the four grayscale frames may enter the policy; numerical
angle, angular velocity, and torque-arrow pixels are excluded.

## Installation

Use Python 3.12. Open a terminal in this directory. Choose either a virtual
environment or Conda; both CPU and NVIDIA GPU execution are supported.

### Option A: Python Virtual Environment

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install one PyTorch build, then install the remaining dependencies:

```bash
# CPU
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
# NVIDIA GPU, Windows or Linux: CUDA 13.0 build
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

```bash
python -m pip install -r requirements.txt
python -m pip check
```

For macOS, follow the official PyTorch installation instructions for your
hardware and use the CPU backend. The CUDA build requires a supported NVIDIA GPU and compatible
driver. Its runtime is bundled with PyTorch; this project does not require
compiling CUDA extensions or installing a separate CUDA Toolkit. Select builds
for other systems using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate dqn-starter
```

Install the appropriate PyTorch build using one of the commands above, then run
`python -m pip check`. PyTorch is installed separately so the CPU/CUDA choice
is explicit. The `pygame-ce` distribution provides the `pygame` import used
by the renderer.

## Smoke Test

Run from this directory after activating the environment:

```bash
python train.py --device cpu --steps 2000 --output runs/smoke
```

A successful run creates logs and checkpoints without exceptions. Zero loss
and random behavior are expected until the TODOs are completed. Check that the
observation is `uint8[4,42,42]`. Use a fresh output directory for each run.

The Primary Network is updated by the optimizer and used to select actions.
The Target Network supplies Bellman target values and is periodically copied
from the Primary Network.

## Training and Evaluation After Completing the TODOs

```bash
python train.py --device auto --steps 150000
python result_show.py runs/training
python run_model.py --episodes 20 --video
python run_model.py --render
python run_model.py --random --episodes 20 --output runs/random_baseline
```

`--device auto` selects CUDA when available, otherwise CPU. You can force
`--device cpu` or `--device cuda`. Default training output is `runs/training`;
default evaluation loads its `best.pt` and writes to `runs/evaluation`.
Specify `--checkpoint` for a nondefault training directory. Do not evaluate
placeholder checkpoints as evidence of a completed assignment.

Validation uses initial-state seeds starting at 10000. Independent evaluation
uses seeds starting at 20000. Use at least 20 independent test episodes, report
mean and standard deviation, and keep test seeds out of checkpoint selection.

Observation images contain only the rod and pivot. Human-facing animation and
video retain the applied-torque arrow. Four frames are ordered oldest to newest,
and the first frame is repeated four times at reset.

## Submission

Follow the assignment manual. Include your completed source, dependency files,
selected checkpoint, configuration, learning logs/plots, independent evaluation
statistics, and representative video. Exclude virtual environments, caches,
and replay-buffer dumps. Record seeds, device, hyperparameters, and run commands.

## References and Attribution

- Mnih et al., Human-level control through deep reinforcement learning, Nature, 2015.
- Gymnasium: https://gymnasium.farama.org/
- Reference implementation: https://github.com/Kchu/DeepRL_PyTorch
- See THIRD_PARTY_NOTICES.txt for the retained renderer license notice.
