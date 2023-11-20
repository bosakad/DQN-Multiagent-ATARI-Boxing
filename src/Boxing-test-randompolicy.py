from pettingzoo.atari import boxing_v2

env = boxing_v2.parallel_env(render_mode="human")
observations, infos = env.reset()

print(observations)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()