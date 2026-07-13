from pathlib import Path
from stable_baselines3 import PPO
from gymnasium_env.envs.kuka_iiwa14 import KukaIiwa14Env

project_root = Path(__file__).resolve().parents[0]

load_path = (
    project_root
    / "models"
    / "ppo_dense_1"
)

n_steps = 10240

env = KukaIiwa14Env(
  render_mode="human", 
  reward_type="dense",
  episode_len=n_steps * 10
  )

model = PPO.load(str(load_path))

obs, info = env.reset()

for i in range(n_steps*10):
  # Get action from the trained model
  action, _ = model.predict(
    obs, 
    deterministic=False,
    )
  
  # Execute the action in the environment
  obs, reward, terminated, truncated, info = env.step(action)

  # Start a new episode if the current one is terminated or truncated
  if terminated or truncated:
    obs, info = env.reset()

env.close()