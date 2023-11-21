from pettingzoo.atari import boxing_v2
import matplotlib.pyplot as plt
from EnvPreprocess import preprocess_boxing

env = boxing_v2.parallel_env(obs_type="grayscale_image", render_mode="human") # remove the render_mode not to render
# env = boxing_v2.parallel_env() 

# preprocess the environment
env = preprocess_boxing(env)

observations, infos = env.reset()

# print(env.action_space("first_0").sample())
# print(env.agents)

print(observations["first_0"].shape)
# actions = {"first_0": 17, "second_0": 0} # test out different actions

i = 0
while env.agents:

    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # actions = {agent: actions[agent] for agent in env.agents} # test out your actions
    
    observations, rewards, terminations, truncations, _ = env.step(actions)

    # show the observation every 100 steps
    # if i % 1000 == 0:

    # obs1 = observations["first_0"]
    # plt.imshow(obs1, cmap="gray")
    # plt.show()

    i += 1

print(i)

env.close()