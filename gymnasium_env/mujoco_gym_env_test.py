from gymnasium_env.envs.kuka_iiwa14 import KukaIiwa14Env
import numpy as np

env = KukaIiwa14Env(render_mode="human")

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation:", obs)
print(obs[21:24])
print(obs[24:27])
print(np.linalg.norm(obs[21:24] - obs[24:27]))

for _ in range(10):
  action = env.action_space.sample()
  
  obs, reward, terminated, truncated, info = env.step(action)

  print(reward)

  env.render()

  if terminated or truncated:
    obs, info = env.reset()

env.close()
