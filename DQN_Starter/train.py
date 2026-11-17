"""Train DQN from scratch using only four grayscale frames."""
import argparse
from collections import deque
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from dqn import DQN, ARCHITECTURE
from environment import ImagePendulum, OBSERVATION_SPEC
from replay_memory import ReplayBuffer
from run_model import evaluate_policy

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/training")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-every", type=int, default=1_000,
                        help="Copy Primary Network to Target Network every N gradient updates")
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--stop-return", type=float, default=None,
                        help="Optionally stop after reaching this greedy validation mean")
    parser.add_argument("--stop-evaluations", type=int, default=2,
                        help="Consecutive successful validations required for early stopping")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "buffer_size", "train_every", "target_every",
                 "epsilon_decay", "log_every", "eval_every", "eval_episodes", "stop_evaluations"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.learning_starts < 0 or args.buffer_size < args.batch_size:
        parser.error("learning-starts must be nonnegative; buffer-size must cover a batch")
    if not 0 <= args.epsilon_end <= 1 or not 0 <= args.gamma <= 1 or args.learning_rate <= 0:
        parser.error("Invalid epsilon, gamma, or learning rate")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "latest.pt").exists() or (args.output / "training.csv").exists():
        parser.error("Output already contains a run. Choose a new --output directory.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    agent = DQN(args.device, args.learning_rate, args.gamma)
    env = ImagePendulum()
    replay = ReplayBuffer(args.buffer_size)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    runtime = {
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(agent.device),
        "gpu": torch.cuda.get_device_name() if agent.device.type == "cuda" else None,
    }
    (args.output / "config.json").write_text(json.dumps(
        {**config, "runtime": runtime, "preprocessing": OBSERVATION_SPEC,
         "architecture": ARCHITECTURE,
         "parameters": sum(parameter.numel() for parameter in agent.primary.parameters())},
        indent=2), encoding="utf-8")
    print(json.dumps(runtime), flush=True)
    print("Observation: uint8[4,42,42]; no angle or angular velocity input", flush=True)
    observation, _ = env.reset(seed=args.seed)
    recent_returns = deque(maxlen=100)
    episode_return = 0.0
    loss = None
    best_return = -float("inf")
    evaluations = []
    successful_evaluations = 0
    should_stop = False
    start = time.perf_counter()
    step = 0
    try:
        with (args.output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "step", "updates", "epsilon", "loss", "mean_100_return", "elapsed_seconds",
            ])
            writer.writeheader()
            for step in range(1, args.steps + 1):
                fraction = min((step - 1) / args.epsilon_decay, 1.0)
                epsilon = 1.0 + fraction * (args.epsilon_end - 1.0)
                action = agent.choose_action(observation, epsilon)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                replay.add(observation, action, reward, next_observation, terminated)
                episode_return += reward
                observation = next_observation
                if terminated or truncated:
                    recent_returns.append(episode_return)
                    episode_return = 0.0
                    observation, _ = env.reset()
                if (step >= args.learning_starts and len(replay) >= args.batch_size
                        and step % args.train_every == 0):
                    loss = agent.learn(replay.sample(args.batch_size))
                    if not np.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}")
                    if agent.updates % args.target_every == 0:
                        agent.sync_target()
                if step % args.eval_every == 0 or step == args.steps:
                    evaluation = evaluate_policy(agent, args.eval_episodes, seed=10_000)
                    evaluation["step"] = step
                    evaluations.append(evaluation)
                    if evaluation["mean_return"] > best_return:
                        best_return = evaluation["mean_return"]
                        agent.save(args.output / "best.pt", step=step, config=config)
                    (args.output / "evaluations.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
                    print(f"Evaluation step={step}: mean_return={evaluation['mean_return']:.2f}", flush=True)
                    if args.stop_return is not None:
                        successful_evaluations = (successful_evaluations + 1
                            if evaluation["mean_return"] >= args.stop_return else 0)
                        should_stop = successful_evaluations >= args.stop_evaluations
                if step % args.log_every == 0 or step == args.steps or should_stop:
                    row = dict(step=step, updates=agent.updates, epsilon=epsilon, loss=loss,
                               mean_100_return=float(np.mean(recent_returns)) if recent_returns else None,
                               elapsed_seconds=round(time.perf_counter() - start, 2))
                    writer.writerow(row)
                    stream.flush()
                    agent.save(args.output / "latest.pt", step=step, config=config)
                    print(json.dumps(row), flush=True)
                if should_stop:
                    print("Validation target reached; stopping early.", flush=True)
                    break
    except KeyboardInterrupt:
        agent.save(args.output / "interrupted.pt", step=step, config=config)
        print("Interrupted; saved interrupted.pt (weights for evaluation, not exact training resume).", flush=True)
        return
    finally:
        env.close()
    summary = {"steps": step, "updates": agent.updates, "best_validation_return": best_return,
               "elapsed_seconds": time.perf_counter() - start, "runtime": runtime,
               "stop_reason": "validation_target" if should_stop else "step_budget"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
