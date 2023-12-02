import supersuit
import warnings

def preprocess_boxing(env, width = 112, height = 147, training=True):
    """
    Preprocess the boxing environment
    :param env: the pettingzoo boxing environment
    :param width: the width of the observation space
    :param height: the height of the observation space

    :return: the preprocessed environment

    NOTE: Final observation space: (height, width, 4 = number of frames stacked)
    """

    # try out other width and height
    width = 80
    height = 98

    with warnings.catch_warnings(): # ignore rendering warnings
        warnings.simplefilter("ignore")

        # force the sticky actions
        if training == True:
            env = supersuit.sticky_actions_v0(env, 0.05)

        # take only 1 color channel - better than grayscale (computationaly)
        env = supersuit.color_reduction_v0(env, mode='G')

        #  as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
        # to deal with frame flickering
        env = supersuit.max_observation_v0(env, 2)

        # downscale observation for faster processing
        env = supersuit.resize_v1(env, int(width), int(height))

        # skip frames for faster processing and less control
        # env = supersuit.frame_skip_v0(env, 1)

        # allow agent to see everything on the screen despite Atari's flickering screen problem
        env = supersuit.frame_stack_v1(env, 4)  

        # reshape the observation space to be compatible with pytorch
        env = supersuit.dtype_v0(env, "float32")


    return env