import taichi as ti
import taichi_gravity

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
# window = ti.ui.Window("N-Body Simulation", (800, 600))
window = ti.ui.Window("N-Body Simulation", (1920, 800))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

# Camera setup
camera.position(0, 0, 10)
camera.lookat(0, 0, 0)

# Control variables
paused = False
reset_requested = False
zoom_sensitivity = 10
camera_pos = 50

taichi_gravity.init_bodies()
i = 0
while window.running:
    # Handle events
    for e in window.get_events():
        if e.key == ti.ui.SPACE:
            paused = not paused
        elif e.key == 'r':
            reset_requested = True
        elif e.key == 'z':
            camera_pos=camera_pos+zoom_sensitivity
        elif e.key == 'x':
            camera_pos=camera_pos-zoom_sensitivity
        elif e.key == 'e':
            break
    
    camera.position(0,0,camera_pos)
    if reset_requested:
        taichi_gravity.init_bodies()
        reset_requested = False
    
    # Update physics
    if not paused:
        taichi_gravity.step()
        
    
    # Render
#    camera.track_user_inputs(window)
    if window.is_pressed(ti.ui.RMB):
        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.4,0.4,0.4))
    scene.particles(positions, radius=0.05, per_vertex_color=colors)
    
    # GUI controls - updated syntax
    with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.2):
        gui.text(f"Bodies: {N:_}")
        gui.text(f"Status: {'Paused' if paused else 'Running'}")
        if gui.button("Reset"):
            reset_requested = True
    
    canvas.scene(scene)
    window.show()
    #window.show(f"frames/frame_{i}.png")
    i=i+1
