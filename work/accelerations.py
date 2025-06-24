import sys
import time
import numpy as np

## Numba
import numba
from numba import njit, prange

import jax.numpy as jnp
import jax

## Taichi
import taichi as ti

# Implement acceleration just using pythonic loops
def get_acceleration_naive_loops(X: np.ndarray) -> np.ndarray:
    acceleration = np.zeros(X.shape)
    for i, ri in enumerate(X):

        sum = np.zeros(3)
        for j, rj in enumerate(X):
            if i == j:
                continue
            diff = rj - ri
            cube = np.linalg.norm(diff) ** 3
            sum += diff / cube

        acceleration[i] = -sum

    return acceleration



# Implement using np arrays and broadcasting
def get_acceleration_numpy(X: np.ndarray) -> np.ndarray:
    vec_diff = X[:, np.newaxis] - X
    distance_matrix = np.linalg.norm(vec_diff, axis=2) ** 3

    # Set distance 0 to inf to avoid dividing by 0
    distance_matrix[distance_matrix == 0] = np.inf

    acceleration = vec_diff / distance_matrix[:, :, np.newaxis]

    return -np.sum(acceleration, axis=0)




@njit
def get_acceleration_naive_loops_numba(X: np.ndarray) -> np.ndarray:
    acceleration = np.zeros(X.shape)
    for i in range(len(X)):

        sum = np.zeros(3)
        for j in range(len(X)):
            if i == j:
                continue
            diff = X[j] - X[i]
            cube = np.linalg.norm(diff) ** 3
            sum = sum + diff / cube

        acceleration[i] = -sum

    return acceleration

# Run one time to make jit compile the code
# print("Numba loops:",validate_acceleration(get_acceleration_naive_loops_numba, get_acceleration_numpy, a))



# Use the same code as above just changing range for prange and setting the njit property
@njit(parallel=True)
def get_acceleration_numba_parallel(X: np.ndarray) -> np.ndarray:
    acceleration = np.zeros(X.shape)
    for i in prange(len(X)):

        sum = np.zeros(3)
        for j in range(len(X)):
            if i == j:
                continue
            diff = X[j] - X[i]
            cube = np.linalg.norm(diff) ** 3
            sum = sum + diff / cube

        acceleration[i] = -sum

    return acceleration

# Run one time to make jit compile the code
# print("Numba parallel:",validate_acceleration(get_acceleration_numba_parallel, get_acceleration_numpy, a))



# @jax.jit # Dont use the annotation to be able to compile for gpu and cpu
def get_acceleration_jax(X: np.ndarray) -> np.ndarray:
    N = len(X)

    def get_i(i):  # Kernel executed in parallel

        vec_diff = X - X[i]
        distance_matrix = jnp.linalg.norm(vec_diff, axis=1) ** 3
        acceleration = vec_diff / distance_matrix[:, jnp.newaxis]

        return -jnp.nansum(acceleration, axis=0)

    # Parallel loop using jax.vmap
    return jax.vmap(get_i)(jnp.arange(N))  # Vectorized version for parallel execution


# Ensure use of 64 bits vs the default of 32 bits
# jax.config.update("jax_enable_x64", True)



# For using with gpu - run in another notebook in colab
# Colab only allows two cores so the parallelization is not effective there.
# from: https://github.com/jax-ml/jax/issues/1598#issuecomment-548031576
get_acceleration_jax_cpu = jax.jit(get_acceleration_jax, backend='cpu')
# get_acceleration_jax_gpu = jax.jit(get_acceleration_jax, backend="gpu")



from jax import lax

@jax.jit # Dont use the annotation to be able to compile for gpu and cpu
def get_acceleration_jax2(X: np.ndarray) -> np.ndarray:
    N = len(X)

    def get_i(i):  # Kernel executed in parallel

        vec_diff = X - X[i]
        
        distance_matrix = jnp.linalg.norm(vec_diff, axis=1) ** 3
        # jax.debug.print("{x}", x=vec_diff.shape)
        acceleration = vec_diff / distance_matrix[:, jnp.newaxis]

        return -jnp.nansum(acceleration, axis=0)

    # Parallel loop using jax.vmap
    return lax.map(get_i, jnp.arange(N))  # Vectorized version for parallel execution




# ti.init(arch=ti.cpu, default_fp=ti.f32)  # Use GPU (or ti.cpu for CPU)
ti.init(arch=ti.gpu, default_fp=ti.f32)  # Use GPU (or ti.cpu for CPU)
# ti.init(arch=ti.gpu)  # Use GPU (or ti.cpu for CPU)
# ti.get_runtime().core.set_capability(ti.core.Capability.vulkan_64bit)


def get_acceleration_taichi(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    n=X.shape[0]

    positions = ti.Vector.field(3, dtype=ti.f32, shape=n)
    acceleration = ti.Vector.field(3, dtype=ti.f32, shape=n)
    
    positions.from_numpy(X)


    @ti.kernel
    def compute_acceleration():
        n = positions.shape[0]
        for i in positions:
            sum_force = ti.math.vec3(0.0)
            for j in range(n):
                if i != j:
                    r = positions[j] - positions[i]
                    r_norm = r.norm()
                    # sum_force += r / (r_norm**3 + 1e-5*r_norm**2)
                    sum_force += r / (r_norm**3 )
            acceleration[i] = -sum_force
    
    compute_acceleration()
    return acceleration.to_numpy()


#-------------------------------


acceleration_functions = [
    get_acceleration_naive_loops,
    get_acceleration_numpy,
    get_acceleration_naive_loops_numba,
    get_acceleration_numba_parallel,
    get_acceleration_jax_cpu,
    get_acceleration_jax2,
    # get_acceleration_jax_gpu
    get_acceleration_taichi,
]

acceleration_functions_dic = {f.__name__.replace("get_acceleration_",""):f   for f in acceleration_functions}

repetitions = 3

if __name__ == "__main__":
    if len(sys.argv) > 2:
        try:
            f_name = str(sys.argv[1])
            N = int(sys.argv[2])

            if f_name not in acceleration_functions_dic:
                print(f"{f_name} not in functions: {acceleration_functions_dic.keys()}")
                sys.exit(1)

            s = 0
            durations = []
            for i in range(3):
                X = np.random.rand(N, 3)
                start_time = time.time()
                result = acceleration_functions_dic[f_name](X)
                if("jax" in  f_name):
                    # Ensure wait to finish
                    result.block_until_ready()
                duration = time.time() - start_time
                durations.append(duration)

            print(np.median(durations))

        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            sys.exit(1)
    else:
        print("Arguments: function and N")
        sys.exit(1)
    