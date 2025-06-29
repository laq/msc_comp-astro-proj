import taichi as ti

ti.init(arch=ti.gpu)
# ti.init(arch=ti.gpu, debug=True)

# # Config
N = int(3e4)  # Change this number as needed
# N = int(2e3)  # Change this number as needed
dt = 0.001
softening = 1e-3

# Fields
# positions = ti.Vector.field(3, dtype=ti.f32, shape=N)
# velocities = ti.Vector.field(3, dtype=ti.f32, shape=N)
positions = ti.Vector.field(3, dtype=ti.f32)
velocities = ti.Vector.field(3, dtype=ti.f32)


def create_fields():
    # Shared root that can be accessed in other modules
    node = ti.root.dense(ti.i, N)
    node.place(positions, velocities)
    return node  # <- return the root so others can use it


@ti.kernel
def init_bodies():
    for i in range(N):
        positions[i] = ti.Vector(
            [ti.random() * 4 - 2, ti.random() * 4 - 2, ti.random() * 4 - 2]
        )
        # velocities[i] = (ti.Vector([ti.random()*0.5-0.25, ti.random()*0.5-0.25, ti.random()*0.5-0.25]))
        velocities[i] = (0, 0, 0)
        # masses[i] = ti.random() * 10 + 1

# @ti.kernel
# def compute_gravity(dt: ti.f32):
#     for i in range(N):
#         force = ti.Vector([0.0, 0.0, 0.0])
#         for j in range(N):
#             if i != j:
#                 r = positions[j] - positions[i]
#                 dist = r.norm() + softening
#                 force += r / (dist**3)
#         velocities[i] += force * dt


@ti.kernel
def update_velocities(dt: ti.f32):
    for i in range(N):
        force = ti.Vector([0.0, 0.0, 0.0])
        for j in range(N):
            if i != j:
                r = positions[j] - positions[i]
                # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
                # This is to prevent 1/0 error which can cause wrong derivative
                dist = r.norm(softening)
                force += r / (dist**3)
        velocities[i] += force * dt


@ti.kernel
def update_positions(dt: ti.f32):
    for i in range(N):
        positions[i] += velocities[i] * dt


def step():
    update_positions(dt / 2)
    update_velocities(dt)
    update_positions(dt / 2)
