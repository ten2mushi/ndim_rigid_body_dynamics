import numpy as np
from rotors_quad_sim import step

class QuadcopterSimulator:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotor
        self.angular_velocity = np.zeros(3)
    
    def update(self, control_inputs: np.ndarray, dt: float):
        """Update quadcopter state for one timestep.
        
        Args:
            control_inputs: Array of 4 motor commands [0-1]
            dt: Timestep duration in seconds
        """
        pos, vel, orient, ang_vel = step(
            self.position.tolist(),
            self.velocity.tolist(),
            self.orientation.tolist(),
            self.angular_velocity.tolist(),
            control_inputs.tolist(),
            dt
        )
        
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self.orientation = np.array(orient)
        self.angular_velocity = np.array(ang_vel)
        
    @property
    def state(self):
        """Get current state as dictionary."""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'orientation': self.orientation,
            'angular_velocity': self.angular_velocity
        }

# Example usage
if __name__ == '__main__':
    sim = QuadcopterSimulator()
    
    # Run simulation for 1 second
    dt = 0.001
    for _ in range(1000):
        # Example hover commands
        controls = np.array([0.5, 0.5, 0.5, 0.5])
        sim.update(controls, dt)
        
    print(f"Final state: {sim.state}")