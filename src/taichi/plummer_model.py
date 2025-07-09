import numpy as np

# Copied from worksheet from Worksheet 7 plummer model notebook.
# initialise the random number generator
rng = np.random.Generator(np.random.PCG64(seed=7897))
# Sample isotropic 3d vectors with a given modulus
def rand_vec3d( mod ):
    N = len(mod)
    phi = 2*np.pi*rng.random(size=N)
    theta = np.arccos( 2*rng.random(size=N)-1 )
    v3d = np.zeros( (N,3) )
    v3d[:,0] = mod * np.cos( phi ) * np.sin( theta )
    v3d[:,1] = mod * np.sin( phi ) * np.sin( theta )
    v3d[:,2] = mod * np.cos( theta )
    # subtract mean
    for i in range(3):
        v3d[:,i] -= np.mean(v3d[:,i])
    return v3d

def sample_plummer(N):
    # number of stars
    # N = 1000

    # particle mass is 1/N
    m = 1/N

    # Sampling the mass, draw radii through inversion sampling from the cumulative mass M
    U = rng.random(size=N)
    rsamp = U**(1/3)/np.sqrt((1-U**(2/3)))

    # create N empty 3D vectors
    x3d = rand_vec3d( rsamp ).astype(np.float32)
    return x3d

# Velocities
def escape_velocity(r):
    """Escape velocity at radius r in Plummer model (in N-body units)."""
    return np.sqrt((2 / np.sqrt(1 + r**2)))


def pq(q):
    """Probability distribution function for q = v / v_e."""
    result = np.zeros_like(q)
    mask = (q > 0) & (q < 1)
    result[mask] = (512 / (7 * np.pi)) * q[mask] ** 2 * (1 - q[mask] ** 2) ** (7 / 2)
    return result


def sample_q(N):
    """Use rejection sampling to draw N samples from p_q(q)."""
    accepted = []
    max_pq = 512 / (7 * np.pi)  # max value of p(q), used for rejection

    while len(accepted) < N:
        q_try = np.random.rand(N)  # Uniform q in [0, 1]
        y_try = np.random.rand(N) * max_pq  # Uniform y in [0, max_pq]

        p_vals = pq(q_try)
        mask = y_try < p_vals
        accepted.extend(q_try[mask])

    return np.array(accepted[:N])


# compute velocity vectors as V = q*ve(r)*r
def sample_velocity_vectors(positions):
    """
    Sample velocities for each star given their positions.
    Returns array of shape (N, 3).
    """
    N = positions.shape[0]
    radii = np.linalg.norm(positions, axis=1)
    v_esc = escape_velocity(radii)

    q_vals = sample_q(N)
    # actual speed for each star
    speeds = q_vals * v_esc

    # Sample random unit directions for each velocity vector
    phi = 2 * np.pi * np.random.rand(N)
    cos_theta = 2 * np.random.rand(N) - 1
    sin_theta = np.sqrt(1 - cos_theta**2)

    vx = speeds * sin_theta * np.cos(phi)
    vy = speeds * sin_theta * np.sin(phi)
    vz = speeds * cos_theta

    velocities = np.vstack((vx, vy, vz)).T.astype(np.float32)
    return velocities


def correct_center_of_mass(positions, velocities):
    # Subtract the center of mass position and velocity to move to the rest frame of the cluster.

    com = np.mean(positions, axis=0)
    com_v = np.mean(velocities, axis=0)

    positions_cm = positions - com
    velocities_cm = velocities - com_v
    return positions_cm, velocities_cm

