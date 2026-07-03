from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium_env.envs.kuka_iiwa14 import KukaIiwa14Env

project_root = Path(__file__).resolve().parents[0]

save_path = str(
  (
    project_root
    / "models"
    / "ppo_kuka_iiwa14"
  )
)

log_path = str(
  (
    project_root
    / "logs" 
  )
)

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

env = KukaIiwa14Env(
  render_mode="human", 
  reward_type="dense",
  episode_len=2048
  )

model = PPO(
  "MlpPolicy", 
  env, 
  verbose=1, 
  tensorboard_log=log_path,
  )

model.learn(
  total_timesteps=100000, 
  callback=TensorboardCallback()
  )

model.save(save_path)

env.close()

