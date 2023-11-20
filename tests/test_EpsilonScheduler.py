import EpsilonScheduler

def _test_scheduler(schedule, step, value, ndigits=5):
    """Tests that the schedule returns the correct value."""
    v = schedule.value(step)
    if not round(v, ndigits) == round(value, ndigits):
        raise Exception(
            f'For step {step}, the scheduler returned {v} instead of {value}'
        )


_schedule = EpsilonScheduler(0.1, 0.2, 3)
_test_scheduler(_schedule, -1, 0.1)
_test_scheduler(_schedule, 0, 0.1)
_test_scheduler(_schedule, 1, 0.141421356237309515)
_test_scheduler(_schedule, 2, 0.2)
_test_scheduler(_schedule, 3, 0.2)
del _schedule

_schedule = EpsilonScheduler(0.5, 0.1, 5)
_test_scheduler(_schedule, -1, 0.5)
_test_scheduler(_schedule, 0, 0.5)
_test_scheduler(_schedule, 1, 0.33437015248821106)
_test_scheduler(_schedule, 2, 0.22360679774997905)
_test_scheduler(_schedule, 3, 0.14953487812212207)
_test_scheduler(_schedule, 4, 0.1)
_test_scheduler(_schedule, 5, 0.1)
del _schedule

print("tests Passed!")