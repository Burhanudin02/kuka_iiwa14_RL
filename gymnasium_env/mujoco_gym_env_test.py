from gymnasium_env.envs.kuka_iiwa14 import KukaIiwa14Env

env = KukaIiwa14Env(render_mode="human")

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation:", obs)

for _ in range(1000):
  action = env.action_space.sample()
  
  obs, reward, terminated, truncated, info = env.step(action)

  env.render()

  if terminated or truncated:
    obs, info = env.reset()

env.close()
