from src.EpsilonScheduler import EpsilonScheduler

import unittest

class TestEpsilonScheduler(unittest.TestCase):

    # helper function to test the scheduler
    def _test_scheduler(self, schedule, step, value, ndigits=5):
        """Tests that the schedule returns the correct value."""
        v = schedule.value(step)
        assert round(v, ndigits) == round(value, ndigits), f'For step {step}, the scheduler returned {v} instead of {value}'

    # test to be called by unittest
    def test_scheduler(self):
        _schedule = EpsilonScheduler(0.1, 0.2, 3)
        self._test_scheduler(_schedule, -1, 0.1)
        self._test_scheduler(_schedule, 0, 0.1)
        self._test_scheduler(_schedule, 1, 0.141421356237309515)
        self._test_scheduler(_schedule, 2, 0.2)
        self._test_scheduler(_schedule, 3, 0.2)
        del _schedule

        _schedule = EpsilonScheduler(0.5, 0.1, 5)
        self._test_scheduler(_schedule, -1, 0.5)
        self._test_scheduler(_schedule, 0, 0.5)
        self._test_scheduler(_schedule, 1, 0.33437015248821106)
        self._test_scheduler(_schedule, 2, 0.22360679774997905)
        self._test_scheduler(_schedule, 3, 0.14953487812212207)
        self._test_scheduler(_schedule, 4, 0.1)
        self._test_scheduler(_schedule, 5, 0.1)
        del _schedule

if __name__ == '__main__':

    unittest.main()