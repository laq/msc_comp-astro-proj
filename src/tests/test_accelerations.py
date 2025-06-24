import numpy as np
from typing import Callable
from  .. import accelerations

import pytest

def assert_acceleration( method1: Callable, method2: Callable, X: np.
ndarray ) -> bool:
    result1 = method1(X)
    result2 = method2(X)
    dtype = np.result_type(result1, result2)

    # Since we are using a mix of f64 and f32 because jax and taichi we reduce the tolerance
    rtol=1e-03
    atol=1e-05


    print(f"dtype:{dtype} Error: {np.sum(result1-result2)}")
    np.testing.assert_allclose(result2, result1, rtol=rtol, atol=atol)



def test_numpy_acc():
    assert_acceleration(accelerations.get_acceleration_naive_loops, accelerations.get_acceleration_numpy, X_64)


N = 100
X_64 = np.random.rand(N,3)
# X_32 = X_64.astype(np.float32)
test_cases = [(fn_name, fn,  X_64) for fn_name, fn in accelerations.acceleration_functions_dic.items()]


@pytest.mark.parametrize("fn_name, fn,  x", test_cases, ids=[str(fn_name) for fn_name,_,_ in test_cases])
def test_acc_fn(fn_name, fn, x):
    print(x.max(), x.min())
    assert_acceleration(fn, accelerations.get_acceleration_numpy, x)
