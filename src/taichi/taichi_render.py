import taichi as ti
import taichi_gravity

ti.init(arch=ti.gpu)
# ti.init(arch=ti.gpu, debug=True)
N = taichi_gravity.N


shared_node = taichi_gravity.create_fields()

positions = taichi_gravity.positions
velocities = taichi_gravity.velocities


colors = ti.Vector.field(4, dtype=ti.f32)
shared_node.place(colors)


@ti.kernel
def init_colors():
    for i in range(N):
        # positions[i] = (ti.Vector([ti.random()*4-2, ti.random()*4-2, ti.random()*4-2]))
        # velocities[i] = (ti.Vector([ti.random()*0.5-0.25, ti.random()*0.5-0.25, ti.random()*0.5-0.25]))
        # masses[i] = ti.random() * 10 + 1
        # colors[i] = ti.Vector([ti.random()/2+0.5, ti.random()/2+0.5, ti.random()/2+0.5])
        colors[i] = ti.Vector([ti.random(), ti.random(), ti.random(), 0.6])


def rotate_vec_right(v: ti.Vector, axis: ti.Vector, angle: ti.f32) -> ti.Vector:
    # To polar
    x = v[0]
    y = v[2]
    r = ti.sqrt(x**2+y**2)
    if r == 0:
        return v
    theta_sin = ti.asin(y/r)
    theta_cos = ti.acos(x/r)

    theta = theta_cos if  theta_sin>=0 else 2*ti.math.pi-theta_cos

    # Angle addition
    theta -= angle

    # To Catesian
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    v[0] = x
    v[2] = y
    return v

 
# Rodrigues rotation
def rotate_vec(v: ti.Vector, axis: ti.Vector, angle: ti.f32) -> ti.Vector:
    axis = axis.normalized()
    cos_a = ti.cos(angle)
    sin_a = ti.sin(angle)
    return (v * cos_a +
            axis.cross(v) * sin_a +
            axis * axis.dot(v) * (1 - cos_a))


def rotate_vec_up(v: ti.Vector, angle: ti.f32) -> ti.Vector:
    i, j = 1,2
    # print(f"Old:{v}")
    # To polar
    x = v[i]
    y = v[j]
    r = ti.sqrt(x**2+y**2)
    if r == 0:
        return v
    theta_sin = ti.asin(y/r)
    theta_cos = ti.acos(x/r)

    theta = theta_cos if  theta_sin>=0 else 2*ti.math.pi-theta_cos

    # Angle addition
    theta -= angle

    # To Catesian
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    
    v[i] = x
    v[j] = y
    # print(f"New:{v}")
    return v



# def rotate_vec:
#     theta_sin = np.arcsin(y)
#     theta_cos = np.arccos(x)

#     theta_sin_deg = np.arcsin(y)/(2*np.pi)*360
#     theta_cos_deg = np.arccos(x)/(2*np.pi)*360

#     theta = theta_cos_deg if  theta_sin_deg>=0 else 360-theta_cos_deg




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

starting_z = 60

camera.position(0, 0, starting_z)
camera.lookat(0, 0, 0)

# Control variables
paused = False
reset_requested = False
zoom_sensitivity = 10
rotation_speed = 0.001

rotate_y = False
rotate_x = 0
record = False


# Axes
axes = ti.Vector.field(3, dtype=ti.f32, shape=6)

# Define origin and unit vectors for X, Y, Z axes
axes[0] = [0, 0, 0]
axes[1] = [10, 0, 0]  # X axis

axes[2] = [0, 0, 0]
axes[3] = [0, 10, 0]  # Y axis

axes[4] = [0, 0, 0]
axes[5] = [0, 0, 10]  # Z axis
# End axes

axes_colors = ti.Vector.field(3, dtype=ti.f32, shape=6)
axes_colors[0] = [1,1,0]
axes_colors[1] = [1,1,0]
axes_colors[2] = [0,1,0]
axes_colors[3] = [0,1,0]
axes_colors[4] = [1,0,0]
axes_colors[5] = [1,0,0]

taichi_gravity.init_bodies_plummer()
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
            rotate_y = not rotate_y
        elif e.key == "v":
            record = True
        elif e.key == "l":
            rotate_x = 1 if not rotate_x else 0
        elif e.key == "k":
            rotate_x = -1 if not rotate_x else 0
        elif e.key == "b":
            record = False

    if zoomed:
        camera.position(*camera_pos)
        print(camera.curr_position)

    if reset_requested:
        taichi_gravity.init_bodies_plummer()
        camera.position(0, 0, starting_z)
        camera.lookat(0, 0, 0)
        reset_requested = False

    if rotate_y:
        # rotate_vew(camera, )
        new_vec = rotate_vec_right(
            camera.curr_position, ti.Vector([0.0, 0.1, 0.0]), rotation_speed
        )
        camera.position(*new_vec)
        camera.lookat(0, 0, 0)

    if rotate_x:
        # rotate_vew(camera, )
        new_vec = rotate_vec_up(
            camera.curr_position, rotation_speed*rotate_x
        )
        camera.position(*new_vec)
        camera.lookat(0, 0, 0)


    # Update physics
    if not paused:
        taichi_gravity.step()

    # Render
    scene.set_camera(camera)
    camera.track_user_inputs(window, hold_key=ti.ui.SHIFT)
    # if window.is_pressed(ti.ui.RMB):
    #     camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
    scene.ambient_light((0.9, 0.9, 0.9))
    scene.particles(positions, radius=0.05, per_vertex_color=colors)
    # scene.particles(positions, radius=0.05, color=(1,130/255,32/255, 0.1))

    # Draw the XYZ axes
    scene.lines(axes, width=2.0, per_vertex_color=axes_colors)  # Red for X
    # scene.lines(axes[2:4], width=2.0, color=(0.0, 1.0, 0.0))  # Green for Y
    # scene.lines(axes[4:6], width=2.0, color=(0.0, 0.5, 1.0))  # Blue for Z
    


    # GUI controls - updated syntax
    with gui.sub_window("Controls", 0.05, 0.05, 0.1, 0.2):
        gui.text(f"Bodies: {N:_}")
        gui.text(f"Status: {'Paused' if paused else 'Running'}")
        if gui.button("Reset"):
            reset_requested = True

    canvas.scene(scene)
    window.show()
    if record:
        window.save_image(f"frames/frame_{i:04d}.png")
    i = i + 1
