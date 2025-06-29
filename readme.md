# Focus project Astrophyicis
## Objective:


## Usage:

## Tests:
* `pytest`
* For specific file `pytest tests/test_integrators.py`


## Make video
Videos are maked frame by frame in the frames folder
Run to make video:  
`ffmpeg -r 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4`
Or to add overlay text 

`ffmpeg -r 15 -i frame_%04d.png -vf "drawtext=text='N-Body simulation 30k stars':fontcolor=white:fontsize=24:x=10:y=10" -c:v libx264 -pix_fmt yuv420p output.mp4`


* Next steps:

[x]  Test acc functions:
    For loop comparing results with loops
* Test integrators
    For loop with test of 3 body problem
    * leapfrog
    * rgk scipy
    Measure Energy
    * Add plummer smooth for close encounters
* Main code:
    * Code for plummer model 
    * Save animation result to file (hd5f?)
* Animation: 
    * Load data
    * Step forward or back
    * Rotate camera speed



Plans for 2 slide presentation:

1st slide purpose of problem:
* Simple one line description
* Plot with max number of stars for 1 or 3 steps?
    * Do memory and time restricted test
    * include different algos for acceleration
    * Including leapfrog?
    * include their energy function??

2nd slide
* use best algorithm to do taichi live simulation
* implement buttons to change the rotation speed of camera





---
Steps:
* Import Leapfrog scheme
* Do animation with Leapfrog scheme
* Do a test of integration with 3body problem
* Set a plummer model to simulate


Done:
[ ] First test of 2d animation with solve_ivp and 3 body problem
Next:
[ ] Organize code to allow easy testing of different methods
[ ] Add my euler or heun solutions
[ ] Add my simplectic solution

[ ] Externalize solver to python file
[ ] Do poc of plummer model moving
[ ] Do poc of 3d 3body problem



--- Did test with taichi using loops and gpu it is not faster than jax, almost the same as numba even with 32 bits

Question: can I make the taichi implementation loop + vector oriented?

Question 2 how high can I go with jax?



### Email:

About my focus project I am considering working on an N-body gravity simulation.
I am thinking I can start animating a cluster created with the Plummer model using a leapfrog scheme
and work my way towards increasing the number of stars.


I have some ideas of additional steps:
1. Parallelize the heavy computation with Numba or Jax(CPU).
2. Do the animation in 3D.
3. Add a hierarchical method to improve scaling.

I am also curious about using a Lagrangian method to incorporate fluid dynamics, but I wonder if that's too much of a stretch.

I appreciate your feedback on my project idea, and would welcome any pointers on how to approach it effectively. 

Best Regards,
Leonardo Quiñonez

