from enum import Enum
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv


class Actions(Enum):
    JOINT_1 = 0
    JOINT_2 = 1
    JOINT_3 = 2
    JOINT_4 = 3
    JOINT_5 = 4
    JOINT_6 = 5
    JOINT_7 = 6


class KukaIiwa14Env(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(
            self, 
            render_mode=None, 
            reward_type="dense", 
            episode_len=1024
            ):

        project_root = Path(__file__).resolve().parents[2]

        model_path = (
            project_root
            / "mjcf"
            / "kuka_iiwa14_scene.xml"
        )

        self.n_joints = 7

        self.EEF_SITE_ID = 2
        self.TARGET_SITE_ID = 0

        self.step_count = 0
        self.max_steps = episode_len

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float64,
        )

        self.reward_type = reward_type

        super().__init__(
            model_path=str(model_path),
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=None,
            width=1280,
            height=720,
        )

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------

    def _get_obs(self):

        qpos = self.data.qpos[: self.n_joints]
        qvel = self.data.qvel[: self.n_joints]
        qacc = self.data.qacc[: self.n_joints]

        eef = self.data.site_xpos[self.EEF_SITE_ID]
        target = self.data.site_xpos[self.TARGET_SITE_ID]

        return np.concatenate(
            [
                qpos,
                qvel,
                qacc,
                eef,
                target,
            ]
        ).astype(np.float64)

    def _get_info(self):

        return {
            "joint_positions": self.data.qpos[: self.n_joints].copy(),
            "joint_velocities": self.data.qvel[: self.n_joints].copy(),
            "joint_accelerations": self.data.qacc[: self.n_joints].copy(),
            "end_effector_position": self.data.site_xpos[self.EEF_SITE_ID].copy(),
            "target_position": self.data.site_xpos[self.TARGET_SITE_ID].copy(),
        }

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset_model(self):

        """
        Called automatically by MujocoEnv.reset().
        """

        self.step_count = 0

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Define the range for randomization
        new_x_pos = np.random.uniform(0.5, 0.6)
        new_y_pos = np.random.uniform(0.5, 0.6)
        new_z_pos = np.random.uniform(0.1, 1.0)

        # Assign the new target position data for observation
        new_target_pos = np.array([
            new_x_pos,
            new_y_pos,
            new_z_pos
        ])

        # Assign the new target site position for visualization
        target_site_pos = np.array([
            new_x_pos,
            new_y_pos,
            new_z_pos
        ])

        # Set the new target position in the simulation
        self.data.site_xpos[self.TARGET_SITE_ID] = new_target_pos
        self.model.site_pos[self.TARGET_SITE_ID] = target_site_pos

        self.set_state(qpos, qvel)

        return self._get_obs()

    # --------------------------------------------------
    # Step
    # --------------------------------------------------

    def step(self, action):

        self.step_count += 1

        # action = np.clip(
        #     action,
        #     self.action_space.low,
        #     self.action_space.high,
        # )

        # normalized action space
        action = np.clip(
            action,
            -1,
            1,
        )

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        reward = self.reward(observation)

        terminated = self._is_done(observation)

        truncated = self.step_count >= self.max_steps

        info = self._get_info()

        # Add the reward to the info dict for logging purpose
        info["reward"] = reward

        info["terminated"] = terminated
        
        info["truncated"] = truncated

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
    
    def reward(self, obs):
        """
        Compute the reward for the current state and action.
        """

        if self.reward_type == "sparse":
            # Sparse reward: +1 if the end-effector is within a threshold distance of the target, else 0
            eef_pos = obs[21:24]  # End-effector position
            target_pos = obs[24:27]  # Target position
            distance = np.linalg.norm(eef_pos - target_pos)
            return 1.0 if distance == 0.0 else 0.0
        
        elif self.reward_type == "dense":
            # Dense reward: negative distance between end-effector and target
            eef_pos = obs[21:24]  # End-effector position
            target_pos = obs[24:27]  # Target position
            distance = np.linalg.norm(eef_pos - target_pos)
            return (1-distance)
        
    def _is_done(self, obs):
        """
        Randomize the target position within a specified range.
        """

        eef_pos = obs[21:24]  # End-effector position
        target_pos = obs[24:27]  # Target position
        distance = np.linalg.norm(eef_pos - target_pos)

        # Indicate an episode is done
        if distance == 0.0: return True 
        
        return False