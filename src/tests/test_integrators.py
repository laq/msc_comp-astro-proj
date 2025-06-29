import pytest
import numpy as np
import time

from .. import integrators
from .. import accelerations

# Use known 3body problem to test stability
# Define initial conditions

r = np.array([[-1, 0], [1, 0], [0, 0]])
r_dot = np.array([[0.080584, 0.588836], [0.080584, 0.588836], [-0.161168, -1.177672]])
time_limit = 21.272338

y = [r, r_dot]
y_flat = np.array(y).flatten()

def calculate_error_position(trajectory):
    return np.sum(np.abs(trajectory[0] - trajectory[-1]))


def test_scipy_integrator():
    start_t = time.time()
    Xs, Vs = integrators.integrator_scipy(
        r, r_dot, fn_acc=accelerations.get_acceleration_numpy, tmax=time_limit
    )
    solve_time = time.time() - start_t
    error_position = calculate_error_position(Xs)
    print(f"Error:{calculate_error_position(Xs)}")
    np.testing.assert_allclose(Xs[-1], Xs[0])








