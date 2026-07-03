from pathlib import Path
from stable_baselines3 import PPO
from gymnasium_env.envs.kuka_iiwa14 import KukaIiwa14Env

project_root = Path(__file__).resolve().parents[0]

save_path = (
    project_root
    / "models"
    / "ppo_kuka_iiwa14"
)

env = KukaIiwa14Env(
  render_mode="human", 
  reward_type="dense"
  )

model = PPO(
  "MlpPolicy", 
  env, 
  verbose=1
  )

model.learn(total_timesteps=1000)

model.save(str(save_path))

env.close()

