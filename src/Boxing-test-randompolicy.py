from pettingzoo.atari import boxing_v2
import matplotlib.pyplot as plt
from EnvPreprocess import preprocess_boxing
import torch

env = boxing_v2.parallel_env(render_mode="human") # remove the render_mode not to render

# preprocess the environment
env = preprocess_boxing(env)


observations, infos = env.reset()


# print(observations["first_0"].shape)
actions = {"first_0": 3, "second_0": 0} # test out different actions

print(env.action_space("first_0"))

i = 0
while env.agents:

    # this is where you would insert your policy
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # actions = {agent: actions[agent] for agent in env.agents} # test out your actions
    

    actions = {"first_0": 10, "second_0": 0} # test out different actions
    observations, rewards, terminations, truncations, _ = env.step(actions)

    actions = {"first_0": 3, "second_0": 0} # test out different actions
    observations, rewards, terminations, truncations, _ = env.step(actions)

    # state = torch.tensor(state[A1], dtype=torch.float32, device=device) # way to convert to tensor
    # state = state.permute(2, 0, 1).unsqueeze(0) # TODO: normalize the image

    # show the observation every x steps
    # if i % 1000 == 20:

        # pass
    
        # uncomment to the observation
    obs1 = observations["first_0"]
    print(obs1.shape)

    # exit(-1)
    # plt.imshow(obs1[:, :, 0], cmap="gray")
    # plt.show()
        # plt.imshow(obs1[:, :, 1], cmap="gray")
        # plt.show()
        # plt.imshow(obs1[:, :, 2], cmap="gray")
        # plt.show()
        # plt.imshow(obs1[:, :, 3], cmap="gray")
        # plt.show()


    i += 1

# number of frames taken
print(i)

env.close()