import argparse
import numpy as np
import mujoco
import imageio

from PPO_grasping_and_holding import SIMPLE_GRASP_XML, SimpleGraspEnv, GraspConfig


def run_random(steps=200, gif_out=None, seed=0):
    env = SimpleGraspEnv(GraspConfig(), render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = []
    for t in range(steps):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if term or trunc:
            obs, info = env.reset()

    if gif_out:
        imageio.mimsave(gif_out, frames, fps=30)
        print(f"Saved simulation to {gif_out}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="Number of random steps")
    parser.add_argument("--gif", type=str, default=None, help="Optional path to save GIF")
    args = parser.parse_args()

    run_random(steps=args.steps, gif_out=args.gif)


if __name__ == "__main__":
    main()
