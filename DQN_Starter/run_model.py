"""Evaluate a saved image-only policy, or a random-action baseline."""
import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from dqn import DQN
from environment import ImagePendulum, TORQUES, OBSERVATION_SPEC

ROOT = Path(__file__).resolve().parent


def evaluate_policy(agent, episodes=5, seed=10_000, episode_length=200,
                    output=None, video=False, render=False):
    env = ImagePendulum(episode_length)
    rng = np.random.default_rng(seed)
    returns = []
    writer = None
    video_writer = None
    action_file = None
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        action_file = (output / "actions.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(action_file)
        writer.writerow(["episode", "step", "action_index", "torque", "reward"])
    try:
        if video and output is not None:
            video_writer = imageio.get_writer(
                output / "policy.mp4", fps=30, codec="libx264", macro_block_size=1,
            )
        if render:
            import pygame
            pygame.display.init()
            screen = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Image-only DQN Pendulum")
            clock = pygame.time.Clock()
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total = 0.0
            for step in range(episode_length):
                if render or (video_writer is not None and episode == 0):
                    frame = env.render()
                    if video_writer is not None and episode == 0:
                        video_writer.append_data(frame)
                    if render:
                        if any(event.type == pygame.QUIT for event in pygame.event.get()):
                            raise KeyboardInterrupt
                        screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
                        pygame.display.flip()
                        clock.tick(30)
                action = (int(rng.integers(len(TORQUES))) if agent is None
                          else agent.choose_action(observation))
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if writer is not None:
                    writer.writerow([episode + 1, step, action, float(TORQUES[action]), reward])
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
        if video_writer is not None:
            video_writer.close()
        if action_file is not None:
            action_file.close()
    return {
        "seed": seed, "episodes": episodes, "episode_length": episode_length,
        "returns": returns, "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "policy": "random" if agent is None else "image_only_dqn",
        "preprocessing": OBSERVATION_SPEC,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/training/best.pt")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.episode_length < 1:
        parser.error("episodes and episode-length must be positive")
    torch.set_num_threads(2)
    agent = None
    checkpoint = None
    if not args.random:
        agent = DQN(args.device)
        checkpoint = agent.load(args.checkpoint)
    result = evaluate_policy(
        agent, args.episodes, args.seed, args.episode_length,
        args.output, args.video, args.render,
    )
    if checkpoint is not None:
        result.update(checkpoint=str(args.checkpoint.resolve()), training_step=checkpoint["step"],
                      device=str(agent.device), architecture=checkpoint["architecture"])
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
