import random
from collections import namedtuple, deque


# define a named tuple to store the transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Cyclic Replay Memory that stores the transitions"""

    def __init__(self, capacity, importanceSampling=False):
        self.memory = deque([], maxlen=capacity)
        self.importanceSampling = importanceSampling

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):

        if not self.importanceSampling:
            return random.sample(self.memory, batch_size) # return random
        
        else:
            # TODO: implement importance sampling
            pass

    def __len__(self):
        return len(self.memory)