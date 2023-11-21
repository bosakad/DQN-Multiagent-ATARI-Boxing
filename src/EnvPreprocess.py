import supersuit

def preprocess_boxing(env):

    # downscale observation for faster processing
    env = supersuit.resize_v1(env, int(112), int(147))

    # skip frames for faster processing and less control
    env = supersuit.frame_skip_v0(env, 4)

    return env