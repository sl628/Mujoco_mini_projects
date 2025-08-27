"""
Minimal PPO-on-MuJoCo example: learn to grasp a cube and lift it above a target height.

Dependencies (one-time):
  pip install mujoco==3.* gymnasium stable-baselines3 numpy imageio matplotlib

Run:
  python simple_grasp_ppo.py --train_steps 200000
  python simple_grasp_ppo.py --eval

Notes:
- The model is intentionally simple (2D X/Z motion + symmetric 2-finger gripper).
- Reward is dense: approach, grasp, lift, and hold.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import imageio


# -----------------------------
# MuJoCo XML (self-contained)
# -----------------------------

SIMPLE_GRASP_XML = r"""
<mujoco model="simple_grasp_2d">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <size nconmax="200" njmax="500" nstack="300000"/>

  <default>
    <geom type="box" rgba="0.7 0.7 0.7 1" condim="4" friction="1.0 0.005 0.0001" solimp="0.8 0.8 0.01" solref="0.02 1"/>
    <joint armature="0.0" damping="2" stiffness="0"/>
    <motor ctrllimited="true" ctrlrange="-1 1" kv="100"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8" width="512" height="512"/>
    <material name="mat_plane" texture="grid" texrepeat="8 8" texuniform="true" reflectance="0.1"/>
  </asset>

  <worldbody>
    <!-- table/ground -->
    <geom name="ground" type="plane" size="2 2 0.1" pos="0 0 0" material="mat_plane"/>

    <!-- cube to grasp -->
    <body name="cube" pos="0.15 0 0.025">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.02 0.02 0.02" density="600" rgba="0.9 0.3 0.3 1"/>
    </body>

    <!-- end-effector body with slide in X and Z -->
    <body name="ee_base" pos="0 0 0.15">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-0.25 0.25"/>
      <joint name="slide_z" type="slide" axis="0 0 1" range="0.05 0.6"/>

      <!-- visual marker -->
      <geom name="ee_marker" type="sphere" size="0.008" rgba="0.1 0.1 0.9 1"/>

      <!-- left finger -->
      <body name="finger_l" pos="0 0.025 0">
        <joint name="finger_l_slide" type="slide" axis="0 1 0" range="-0.01 0.05"/>
        <geom name="finger_l_geom" type="box" size="0.01 0.005 0.03" pos="0 0 0" rgba="0.2 0.2 0.8 1"/>
      </body>

      <!-- right finger -->
      <body name="finger_r" pos="0 -0.025 0">
        <joint name="finger_r_slide" type="slide" axis="0 1 0" range="-0.05 0.01"/>
        <geom name="finger_r_geom" type="box" size="0.01 0.005 0.03" pos="0 0 0" rgba="0.2 0.2 0.8 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- EE planar motion -->
    <motor name="ax" joint="slide_x" gear="1"/>
    <motor name="az" joint="slide_z" gear="1"/>

    <!-- gripper fingers (independent, we'll command symmetrically in code) -->
    <motor name="grip_l" joint="finger_l_slide" gear="1"/>
    <motor name="grip_r" joint="finger_r_slide" gear="1"/>
  </actuator>

  <sensor>
    <jointpos name="jx" joint="slide_x"/>
    <jointpos name="jz" joint="slide_z"/>
    <jointpos name="jgr_l" joint="finger_l_slide"/>
    <jointpos name="jgr_r" joint="finger_r_slide"/>
    <framepos name="cube_pos" objtype="body" objname="cube"/>
  </sensor>
</mujoco>
"""


# -----------------------------
# Environment
# -----------------------------

@dataclass
class GraspConfig:
    ctrl_rate_hz: int = 50  # control updates per second
    sim_substeps: int = 5   # mj_step per control step
    horizon: int = 400
    target_height: float = 0.20
    hold_required_steps: int = 30
    ee_x_range: tuple = (-0.20, 0.25)
    ee_z_range: tuple = (0.05, 0.55)
    grip_open_close_speed: float = 0.015  # per control step (in joint units)
    randomize: bool = True  # randomize cube position and friction per episode


class SimpleGraspEnv(gym.Env):
    """
    2D EE (X, Z) + 2-finger gripper. Task: grasp cube and lift above target height, then hold.
    Observation (17 floats):
      - ee x, z, cube x, y, z, cube linvel (3), cube angvel (3)
      - gripper opening (scalar), finger_l pos, finger_r pos
      - delta xz to cube (2), height above table (1)
    Action (3 floats):
      - dx, dz (end-effector velocity commands), dgrip (open(+)/close(-))
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: GraspConfig = GraspConfig(), render_mode=None):
        super().__init__()
        self.cfg = cfg

        self.model = mujoco.MjModel.from_xml_string(SIMPLE_GRASP_XML)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.ctrl_dt = 1.0 / cfg.ctrl_rate_hz

        # Build spaces
        high_obs = np.ones(17, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.viewer = None

        self.t = 0
        self.hold_counter = 0

        # Cache ids
        self.jid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
        self.jid_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")
        self.jid_gl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_l_slide")
        self.jid_gr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_r_slide")

        self.aid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ax")
        self.aid_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "az")
        self.aid_gl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_l")
        self.aid_gr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_r")

        self.sid_cube_pos = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_pos")

        # Useful addresses
        self.qpos0 = self.data.qpos.copy()

    # ---------------------
    # Core API
    # ---------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0
        self.hold_counter = 0

        # Randomize cube XY and friction a bit each episode
        if self.cfg.randomize:
            # position: random x in [0.05, 0.20], small y jitter
            cx = self.np_random.uniform(0.05, 0.20)
            cy = self.np_random.uniform(-0.03, 0.03)
            cz = 0.025
            self._set_body_pos("cube", [cx, cy, cz])

            # random friction on cube geom
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            fric = self.model.geom_friction[gid]
            fric[0] = float(self.np_random.uniform(0.3, 1.2))  # sliding
            self.model.geom_friction[gid] = fric

        # Reset EE to middle, gripper open
        self._set_joint_pos("slide_x", 0.0)
        self._set_joint_pos("slide_z", 0.20)
        self._set_joint_pos("finger_l_slide", 0.04)
        self._set_joint_pos("finger_r_slide", -0.04)

        # minor settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Interpret actions as *velocity commands* for X, Z and gripper aperture
        dx = float(action[0]) * 0.01  # meters per ctrl step
        dz = float(action[1]) * 0.01
        dgrip = float(action[2]) * self.cfg.grip_open_close_speed

        # current positions
        x = self._get_joint_pos(self.jid_x)
        z = self._get_joint_pos(self.jid_z)
        gl = self._get_joint_pos(self.jid_gl)
        gr = self._get_joint_pos(self.jid_gr)

        # target positions
        x_t = np.clip(x + dx, *self.cfg.ee_x_range)
        z_t = np.clip(z + dz, *self.cfg.ee_z_range)
        gl_t = np.clip(gl + dgrip, -0.005, 0.05)
        gr_t = np.clip(gr - dgrip, -0.05, 0.005)  # opposite sign

        # convert to actuator targets (simple P controller via position motors)
        self.data.ctrl[self.aid_x] = self._pos_to_motor(x_t, x)
        self.data.ctrl[self.aid_z] = self._pos_to_motor(z_t, z)
        self.data.ctrl[self.aid_gl] = self._pos_to_motor(gl_t, gl)
        self.data.ctrl[self.aid_gr] = self._pos_to_motor(gr_t, gr)

        # simulate
        for _ in range(self.cfg.sim_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, grasped, lifted = self._compute_reward(obs)

        self.t += 1
        terminated = False
        truncated = self.t >= self.cfg.horizon

        # success condition: lifted above target for N consecutive steps
        if lifted:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        success = self.hold_counter >= self.cfg.hold_required_steps
        if success:
            reward += 10.0
            terminated = True

        info = {"grasped": grasped, "lifted": lifted, "success": success}
        return obs, reward, terminated, truncated, info

    # ---------------------
    # Helpers
    # ---------------------
    def _get_obs(self):
        # ee positions (joint positions are absolute for slides)
        x = self._get_joint_pos(self.jid_x)
        z = self._get_joint_pos(self.jid_z)
        gl = self._get_joint_pos(self.jid_gl)
        gr = self._get_joint_pos(self.jid_gr)

        # cube pose & velocities
        cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_x, cube_y, cube_z = self.data.xpos[cube_bid].copy()
        cube_linvel = self.data.cvel[cube_bid][3:6].copy()  # world linear vel
        cube_angvel = self.data.cvel[cube_bid][0:3].copy()

        # derived
        grip_opening = (gl - gr) * 0.5  # approx aperture (symmetry)
        dx = (cube_x - x)
        dz = (cube_z - z)

        obs = np.array([
            x, z,
            cube_x, cube_y, cube_z,
            cube_linvel[0], cube_linvel[1], cube_linvel[2],
            cube_angvel[0], cube_angvel[1], cube_angvel[2],
            grip_opening, gl, gr,
            dx, dz,
            cube_z  # height
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, obs):
        # unpack
        x, z = obs[0], obs[1]
        cube_x, cube_y, cube_z = obs[2], obs[3], obs[4]
        grip_opening = obs[11]
        dx, dz = obs[14], obs[15]

        # shaping terms
        dist = math.sqrt(dx * dx + dz * dz + (cube_y ** 2))  # EE to cube distance
        approach_reward = 1.0 * (1.0 - np.tanh(3.0 * dist))

        # encourage being above the cube when close in x
        above_bonus = 0.2 if (abs(dx) < 0.03 and z > cube_z + 0.02) else 0.0

        # gentle penalty for very open gripper when close to cube
        grip_penalty = 0.0
        if dist < 0.05 and grip_opening > 0.03:
            grip_penalty = -0.05 * (grip_opening - 0.03) * 100.0

        # detect grasp (cube above table and close to EE vertically, reduces distance)
        grasped = (cube_z > 0.03) and (dist < 0.06)

        # lifting reward
        target = self.cfg.target_height
        lift_reward = 2.0 * max(0.0, cube_z - 0.03)  # any lifting above table
        big_lift_bonus = 3.0 if cube_z > target else 0.0

        reward = approach_reward + above_bonus + grip_penalty + lift_reward + big_lift_bonus
        lifted = cube_z > target
        return reward, grasped, lifted

    def _pos_to_motor(self, target, current):
        # position motor: drive toward target; residual -> control
        return float(np.clip(target - current, -1.0, 1.0))

    def _get_joint_pos(self, joint_id):
        adr = self.model.jnt_qposadr[joint_id]
        return float(self.data.qpos[adr])

    def _set_joint_pos(self, name, value):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr] = value

    def _set_body_pos(self, name, pos3):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.data.xpos[bid] = np.array(pos3)  # will be overwritten by mj_forward
        # modify qpos for freejoint if present
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        if jid != -1:
            adr = self.model.jnt_qposadr[jid]
            # freejoint: 7D (pos[3], quat[4])
            self.data.qpos[adr:adr+3] = pos3
            self.data.qpos[adr+3:adr+7] = np.array([1, 0, 0, 0], dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        # headless RGB render using mjv/mjr
        width, height = 480, 360
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scn = mujoco.MjvScene(self.model, 2000)
        con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        cam.lookat[:] = [0.08, 0.0, 0.12]
        cam.distance = 0.5
        cam.azimuth = 90
        cam.elevation = -25

        mujoco.mjv_updateScene(self.model, self.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scn, con)
        mujoco.mjr_readPixels(rgb, depth, mujoco.MjrRect(0, 0, width, height), con)
        mujoco.mjr_freeContext(con)
        scn.free()
        return np.flipud(rgb)  # flip vertical


# -----------------------------
# Training / Evaluation
# -----------------------------

def make_env(seed=0, record_monitor=True):
    env = SimpleGraspEnv(GraspConfig())
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env.cfg.horizon)
    if record_monitor:
        env = Monitor(env)
    env.reset(seed=seed)
    return env

def train(train_steps=200_000, save_path="ppo_simple_grasp.zip", seed=0):
    env = make_env(seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        n_epochs=10,
        device="auto",
        tensorboard_log="./tb_simple_grasp",
    )
    model.learn(total_timesteps=train_steps, progress_bar=True)
    model.save(save_path)
    env.close()
    print(f"Saved model to: {save_path}")

def evaluate(model_path="ppo_simple_grasp.zip", episodes=3, gif_out="eval.gif", seed=123):
    env = make_env(seed, record_monitor=False)
    model = PPO.load(model_path, device="auto")

    frames = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_frames = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            frame = env.render()
            ep_frames.append(frame)
        frames += ep_frames

    imageio.mimsave(gif_out, frames, fps=30)
    env.close()
    print(f"Wrote evaluation GIF: {gif_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--eval", action="store_true", help="Only run evaluation using saved model")
    parser.add_argument("--model", type=str, default="ppo_simple_grasp.zip")
    args = parser.parse_args()

    if args.eval:
        evaluate(model_path=args.model)
    else:
        train(train_steps=args.train_steps, save_path=args.model)
        evaluate(model_path=args.model)

if __name__ == "__main__":
    main()
