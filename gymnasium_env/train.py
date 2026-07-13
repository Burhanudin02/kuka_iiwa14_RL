from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium_env.envs.kuka_iiwa14 import KukaIiwa14Env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Configure paths for saving models and logs
project_root = Path(__file__).resolve().parents[0]

save_path = str(
  (
    project_root
    / "models"
    / "ppo_dense_6"
  )
)

log_path = str(
  (
    project_root
    / "logs" 
  )
)

# Tensorboard callback to log additional metrics during training
class TensorboardCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)

  def _on_step(self) -> bool:
    # Log additional metrics to TensorBoard
    infos = self.locals["infos"]

    for info in infos:
      self.logger.record("episode/reward", info["reward"])
      self.logger.record("episode/terminated", info["terminated"])
      self.logger.record("episode/truncated", info["truncated"])

    return True

def main(save_path, log_path):

  # Training environment parameters
  # rollout_size = n_steps * n_envs
  # Bigger n_steps, better agent understanding 
  # of the environment, but slower training
  n_steps = 10240
  n_envs = 8
  
  env = make_vec_env(
    KukaIiwa14Env,
    n_envs=n_envs,
    env_kwargs={
      "render_mode": None,
      "reward_type": "dense",
      "episode_len": n_steps*10,
    },
    vec_env_cls=SubprocVecEnv,
  )
  
  # env = KukaIiwa14Env(
  #   render_mode="human", 
  #   reward_type="sparse",
  #   episode_len=10 * n_steps
  #   )
  
  # model = PPO(
  #   "MlpPolicy", 
  #   env, 
  #   verbose=1, 
  #   tensorboard_log=log_path,
  #   n_steps=n_steps,
  #   )
  
  model = PPO.load(
    "/home/host-20-04/mujoco_workspace/kuka_iiwa14_RL/gymnasium_env/models/ppo_dense_5.zip",
    env=env,
  )
  
  model.learn(
    total_timesteps=1000*n_steps, 
    callback=TensorboardCallback()
    )
  
  model.save(save_path)
  
  env.close()

if __name__ == "__main__":
  main(save_path, log_path)