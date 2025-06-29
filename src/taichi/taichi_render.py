import taichi as ti
import taichi_gravity
import star_colors

ti.init(arch=ti.gpu)
# ti.init(arch=ti.gpu, debug=True)
N = taichi_gravity.N


@ti.kernel
def init_colors():
    for i in range(N):
        # positions[i] = (ti.Vector([ti.random()*4-2, ti.random()*4-2, ti.random()*4-2]))
        # velocities[i] = (ti.Vector([ti.random()*0.5-0.25, ti.random()*0.5-0.25, ti.random()*0.5-0.25]))
        # masses[i] = ti.random() * 10 + 1
        # colors[i] = ti.Vector([ti.random()/2+0.5, ti.random()/2+0.5, ti.random()/2+0.5])
        colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])


def cartesian_to_spherical_camera(camera_cartesian):
    x, y, z = camera_cartesian
    r = ti.sqrt(x * x + y * y + z * z)
    theta = ti.acos(z / r) if r > 0 else 0.0  # polar angle
    phi = ti.atan2(y, x)                      # azimuth
    camera_spherical = ti.Vector([r, theta, phi])
    return camera_spherical



def spherical_to_cartesian_camera(camera_spherical):
    r, theta, phi = camera_spherical
    x = r * ti.sin(theta) * ti.cos(phi)
    y = r * ti.sin(theta) * ti.sin(phi)
    z = r * ti.cos(theta)
    camera_cartesian = ti.Vector([x, y, z])
    return camera_cartesian


def rotate_vew(camera):
    camera_cartesian = camera.curr_position
    camera_spherical = cartesian_to_spherical_camera(camera_cartesian)
    print("Cameras")
    print("Spherical",camera_spherical)
    camera_spherical[1]+=0.1
    camera_cartesian = spherical_to_cartesian_camera(camera_spherical)
    print("Cartesian",camera_cartesian)
    camera.position(*camera_cartesian)


def rotate_vec(v: ti.Vector, axis: ti.Vector, angle: ti.f32) -> ti.Vector:
    axis = axis.normalized()
    cos_a = ti.cos(angle)
    sin_a = ti.sin(angle)
    return (v * cos_a +
            axis.cross(v) * sin_a +
            axis * axis.dot(v) * (1 - cos_a))



# def init_colors():
#     colors_np = star_colors.generate_star_colors(N)
#     print(colors_np.shape, colors.shape, colors_np[:4])
#     colors.from_numpy(colors_np)
#     # print(colors[:10])


# Fields
# positions = ti.Vector.field(3, dtype=ti.f32, shape=N)
# velocities = ti.Vector.field(3, dtype=ti.f32, shape=N)
# colors = ti.Vector.field(3, dtype=ti.f32, shape=N)

shared_node = taichi_gravity.create_fields()

positions = taichi_gravity.positions
velocities = taichi_gravity.velocities


colors = ti.Vector.field(3, dtype=ti.f32)
shared_node.place(colors)

init_colors()


# Initialize GUI
# window = ti.ui.Window("N-Body Simulation", (400, 150))
# window = ti.ui.Window("N-Body Simulation", (800, 600))
window = ti.ui.Window("N-Body Simulation", (1920, 800))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

# Camera setup
# Set initial camera position
camera_pos = ti.Vector([0.0, 0.0, 10.0])

starting_z = 40

camera.position(0, 0, starting_z)
camera.lookat(0, 0, 0)

# Control variables
paused = False
reset_requested = False
zoom_sensitivity = 10

rotate = True

taichi_gravity.init_bodies()
i = 0
while window.running:
    # Handle events
    zoomed = False
    for e in window.get_events():
        if e.key == ti.ui.SPACE:
            paused = not paused
        elif e.key == "r":
            reset_requested = True
        elif e.key == "z":
            camera_pos = camera_pos + zoom_sensitivity
            zoomed = True
        elif e.key == "x":
            camera_pos = camera_pos - zoom_sensitivity
            zoomed = True
        elif e.key == "p":
            rotate = False
            print("Cartesian",camera.curr_position)
        elif e.key == "e":
            break

    if zoomed:
        camera.position(*camera_pos)
        print(camera.curr_position)

    if reset_requested:
        taichi_gravity.init_bodies()
        camera.position(0, 0, starting_z)
        camera.lookat(0, 0, 0)
        reset_requested = False

    if rotate:
        # rotate_vew(camera, )
        new_vec = rotate_vec(camera.curr_position, ti.Vector([0.0,0.1,0.0]), 0.001)
        camera.position(*new_vec)        
        camera.lookat(0,0,0)


    # Update physics
    if not paused:
        taichi_gravity.step()

    # Render
    scene.set_camera(camera)
    camera.track_user_inputs(window, hold_key=ti.ui.SHIFT)
    # if window.is_pressed(ti.ui.RMB):
    #     camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.particles(positions, radius=0.05, per_vertex_color=colors)
    # scene.particles(positions, radius=0.05, color=(1,130/255,32/255))

    # GUI controls - updated syntax
    with gui.sub_window("Controls", 0.05, 0.05, 0.1, 0.2):
        gui.text(f"Bodies: {N:_}")
        gui.text(f"Status: {'Paused' if paused else 'Running'}")
        if gui.button("Reset"):
            reset_requested = True

    canvas.scene(scene)
    window.show()
    # window.show(f"frames/frame_{i}.png")
    i = i + 1
