# Focus project Astrophyicis
## Report:
See [report](report/report.pdf)

## How to run
1. Install requirements: `pip install -r requirements.txt`
    * Vulkan might be required to run Taichi efficiently
2. Set Number of stars: Set `taichi_gravity.py` N value
3. Run: `python taichi_render.py`
4. Close the window to exit

## How to control the simulation
The simulation has quirky controls configured in `taichi_renderer`.

* "Space": hold to pause
* "WASD": press to move the camera videogame style
* "Shift": Leave press to make the camera follow the mouse
* "P": Orbit the center with the camera on the Y axis (Right)
* "l","k": Orbit the center with the camera on the X axis (Up and Down respectively)
* "v": press to start recording every frame.  (Frames would be saved as png in frames folder)
* "b": press to stop recording

## How to make a video
Videos are maked frame by frame in the frames folder
Run to make video:  
`ffmpeg -r 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4`
Or to add overlay text 

`ffmpeg -r 15 -i frame_%04d.png -vf "drawtext=text='N-Body simulation 40k stars':fontcolor=white:fontsize=24:x=10:y=10" -c:v libx264 -pix_fmt yuv420p output.mp4`


## Tests:
Some simple tests to ensure the acceleration implementations are returning the same result.
* `pytest`
* For specific file `pytest tests/test_accelerations.py`
