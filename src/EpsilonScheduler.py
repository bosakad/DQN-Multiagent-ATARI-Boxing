import numpy as np

class EpsilonScheduler:
    def __init__(self, EPS_START, EPS_END, EPS_DECAY):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param EPS_START: initial value
        :param EPS_END: final value
        :param EPS_DECAY: decay rate
        """

        self.EPS_END = EPS_END
        self.EPS_START = EPS_START
        self.EPS_DECAY = EPS_DECAY

        self.a = EPS_END + (EPS_START - EPS_END)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        :param step:  The step at which to compute the interpolation.
        :rtype: float.  The interpolated value.
        """

        eps_threshold =  self.a * np.exp(-1. * step / self.EPS_DECAY)
        
        return eps_threshold

