# utils/rotors.rs:
Implementation of "N-Dimensional Rigid Body Dynamics" (https://web.engr.oregonstate.edu/~mjb/cs550/Projects/Papers/RigidBodyDynamics.pdf)

# utils/quadcopter.rs
Rotors and Bivectors used to a create a simple quadcopter simulator (instead of using quat)

# lib.rs
Quadcopter simulator as a python package which can be installed in Blender env

# download repository then
python -m venv venv
source venv/bin/activate
pip install maturin
pip install numpy

# Build new wheel
maturin build --release --out dist

# Install to Blender's Python (with sudo) (make sure venv python and blender python are both same version)
sudo /Applications/Blender.app/Contents/Resources/4.3/python/bin/python3.11 -m pip install --force-reinstall dist/rotors_quad_sim-0.1.0-cp311-cp311-macosx_11_0_arm64.whl
(blender python path + wheel build path)


# in a blender script:

```python
import site
import sys
sys.path.append(site.getusersitepackages())
import rotors_quad_sim

quad = rotors_quad_sim.QuadcopterSim()
pos, vel, orient, ang_vel = quad.step([thurst1, thurst2, thurst3, thurst4], dt)
```

