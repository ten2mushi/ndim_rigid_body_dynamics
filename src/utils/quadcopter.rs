use std::f64::consts::PI;
use std::fmt::Debug;
use rustfft::num_traits::Float;

use crate::utils::rotors::{
    Rotor, Rotor3D, SymplecticIntegrator, Vector, Bivector, RigidBody, InertialTensor
};
use crate::utils::constants::{GRAVITY, AIR_DENSITY, VISCOSITY};
use std::ops::Add;

pub trait Two {
    fn two() -> Self;
}

impl Two for f32 {
    #[inline]
    fn two() -> Self { 2.0 }
}

impl Two for f64 {
    #[inline]
    fn two() -> Self { 2.0 }
}

#[derive(Clone, Debug)]
pub struct MotorConfig {
    pub prop_radius: f64,      // m
    pub prop_pitch: f64,       // m
    pub max_rpm: f64,         
    pub moment_of_inertia: f64, // kg·m²
    pub thrust_coefficient: f64,  // N/(rad/s)²
    pub torque_coefficient: f64,  // N·m/(rad/s)²
    pub drag_coefficient: f64,    // N·s²/m²
    pub kv_rating: f64,         // rpm/V
    pub resistance: f64,        // Ω
    pub voltage: f64,           // V
}

impl Default for MotorConfig {
    fn default() -> Self {
        Self {
            prop_radius: 0.127,  // 5-inch prop
            prop_pitch: 0.0762,  // 3-inch pitch
            max_rpm: 12000.0,
            moment_of_inertia: 2e-5,
            thrust_coefficient: 1.91e-6,
            torque_coefficient: 2.75e-7,
            drag_coefficient: 0.012,
            kv_rating: 2300.0,
            resistance: 0.1,
            voltage: 14.8,  // 4S LiPo
        }
    }
}

/// State of a single motor
#[derive(Clone, Debug)]
pub struct Motor {
    config: MotorConfig,
    // Fixed position relative to COM
    position: Vector<f64, 3>,
    pub rpm: f64,
    pub current: f64,
    pub temperature: f64,
    // Direction of rotation (+1 CCW, -1 CW)
    spin_direction: f64,
}

impl Motor {
    pub fn new(position: Vector<f64, 3>, spin_direction: f64, config: MotorConfig) -> Self {
        Self {
            config,
            position,
            rpm: 0.0,
            current: 0.0,
            temperature: 20.0,
            spin_direction,
        }
    }
    
    pub fn set_throttle(&mut self, throttle: f64) {
        let throttle = throttle.clamp(0.0, 1.0);
        let voltage = throttle * self.config.voltage;
        self.rpm = voltage * self.config.kv_rating;
        
        let back_emf = self.rpm / self.config.kv_rating;
        self.current = (voltage - back_emf) / self.config.resistance;
    }
    
    pub fn compute_thrust(&self) -> Vector<f64, 3> {
        let angular_velocity = self.rpm * 2.0 * PI / 60.0;  // convert to rad/s
        let thrust = self.config.thrust_coefficient * angular_velocity.powi(2);
        Vector::new([0.0, 0.0, thrust])
    }
    
    pub fn compute_reactive_torque(&self) -> Bivector<f64, 3> {
        let angular_velocity = self.rpm * 2.0 * PI / 60.0;
        let torque = self.spin_direction * 
                    self.config.torque_coefficient * 
                    angular_velocity.powi(2);
                    
        Bivector::new(&[0.0, 0.0, torque]).unwrap()
    }
    
    pub fn compute_thrust_torque(&self) -> Bivector<f64, 3> {
        Bivector::from_outer_product(&self.position, &self.compute_thrust())
    }
}

#[derive(Clone, Debug)]
pub struct QuadcopterConfig {
    pub mass: f64,             // kg
    pub arm_length: f64,       // m
    pub height: f64,           // m
    pub drag_coefficient: f64,  // N·s²/m²
    
    pub motor_config: MotorConfig,
    
    pub orientation_kp: f64,
    pub orientation_ki: f64,
    pub orientation_kd: f64,
    
    pub position_kp: f64,
    pub position_ki: f64,
    pub position_kd: f64,
}

impl Default for QuadcopterConfig {
    fn default() -> Self {
        Self {
            mass: 1.0,
            arm_length: 0.2,
            height: 0.1,
            drag_coefficient: 0.47,
            motor_config: MotorConfig::default(),
            orientation_kp: 5.0,
            orientation_ki: 0.0,
            orientation_kd: 0.5,
            position_kp: 2.0,
            position_ki: 0.0,
            position_kd: 0.3,
        }
    }
}

pub struct Quadcopter {
    config: QuadcopterConfig,
    pub body: RigidBody<f64, 3>,
    motors: [Motor; 4],
    orientation_error_integral: Bivector<f64, 3>,
    position_error_integral: Vector<f64, 3>,
}

impl Quadcopter {
    pub fn new(config: QuadcopterConfig) -> Self {
        let motor_positions = [
            Vector::new([ config.arm_length,  config.arm_length, 0.0]), // Front right
            Vector::new([ config.arm_length, -config.arm_length, 0.0]), // Back right 
            Vector::new([-config.arm_length, -config.arm_length, 0.0]), // Back left
            Vector::new([-config.arm_length,  config.arm_length, 0.0]), // Front left
        ];
        
        let motors = [
            Motor::new(motor_positions[0].clone(), -1.0, config.motor_config.clone()),
            Motor::new(motor_positions[1].clone(),  1.0, config.motor_config.clone()),
            Motor::new(motor_positions[2].clone(), -1.0, config.motor_config.clone()),
            Motor::new(motor_positions[3].clone(),  1.0, config.motor_config.clone()),
        ];
        
        let mut inertia = InertialTensor::<f64, 3>::zero();
        
        let arm_mass = config.mass * 0.2;  // 20% of mass in arms
        let arm_inertia = arm_mass * config.arm_length.powi(2) / 3.0;
        
        inertia.set(0, 0, arm_inertia);  // xy plane
        inertia.set(1, 1, arm_inertia);  // xz plane
        inertia.set(2, 2, 2.0 * arm_inertia);  // yz plane
        
        for motor in &motors {
            let r_squared = motor.position.magnitude_squared();
            let motor_mass = config.mass * 0.1;  // 10% of mass per motor
            for i in 0..3 {
                inertia.set(i, i, inertia.get(i, i) + motor_mass * r_squared);
            }
        }
        
        let mut inv_inertia = InertialTensor::<f64, 3>::zero();
        for i in 0..3 {
            inv_inertia.set(i, i, 1.0 / inertia.get(i, i));
        }
        
        let body = RigidBody::new(
            Vector::new([0.0, 0.0, 0.0]),  // initial position
            config.mass,
            inertia,
            inv_inertia
        );
        
        Self {
            config,
            body,
            motors,
            orientation_error_integral: Bivector::zero(),
            position_error_integral: Vector::zero(),
        }
    }
    
    pub fn update(&mut self, dt: f64, motor_commands: &[f64; 4]) {
        for (motor, &command) in self.motors.iter_mut().zip(motor_commands) {
            motor.set_throttle(command);
        }
        
        let mut net_force = Vector::new([0.0, 0.0, -GRAVITY * self.config.mass]);
        let mut net_torque = Bivector::zero();
        
        for motor in &self.motors {
            let thrust = self.body.state.orientation.rotate(&motor.compute_thrust());
            net_force = net_force + thrust;
            
            net_torque = net_torque + 
                        motor.compute_thrust_torque() + 
                        motor.compute_reactive_torque();
        }
        
        let velocity = &self.body.state.velocity;
        let drag = velocity.scale(
            -self.config.drag_coefficient * 
            velocity.magnitude() * 
            AIR_DENSITY
        );
        net_force = net_force + drag;
        
        self.body.step(dt, &net_force, &net_torque);
    }
    
    pub fn compute_attitude_control(
        &mut self,
        target_orientation: &Rotor<f64, 3>,
        dt: f64
    ) -> [f64; 4] {
        // in test
        let error = target_orientation.compose(&self.body.state.orientation.reverse());
        
        let (axis, angle) = error.to_axis_angle();
        let error_bivector = axis.scale(angle);
        
        self.orientation_error_integral = 
            &self.orientation_error_integral + 
            &error_bivector.scale(dt);
        
        let control_bivector = 
            &(&error_bivector.scale(self.config.orientation_kp) + 
              &self.orientation_error_integral.scale(self.config.orientation_ki)) - 
            &self.body.state.angular_velocity.scale(self.config.orientation_kd);
        
        let control_rotor = control_bivector.to_rotor();
        let (roll, pitch, yaw) = control_rotor.to_euler_angles();
        
        let mut commands = [0.0; 4];
        
        // hovering thrust
        let base_thrust = 0.318;
        
        commands[0] = base_thrust - pitch - roll + yaw;  // Front right
        commands[1] = base_thrust - pitch + roll - yaw;  // Back right
        commands[2] = base_thrust + pitch + roll + yaw;  // Back left
        commands[3] = base_thrust + pitch - roll - yaw;  // Front left
        
        commands.iter_mut().for_each(|c| *c = c.clamp(0.0, 1.0));
        
        commands
    }
    
    pub fn compute_position_control(
        &mut self,
        target_position: &Vector<f64, 3>,
        dt: f64
    ) -> Rotor3D<f64> {
        // in test
        let error = target_position - &self.body.state.position;
        
        self.position_error_integral = self.position_error_integral.clone() + 
                                     error.scale(dt);
        
        let desired_acceleration = error.scale(self.config.position_kp) +
                                 self.position_error_integral.scale(self.config.position_ki) -
                                 self.body.state.velocity.scale(self.config.position_kd);
        
        let thrust_dir = desired_acceleration.normalized().unwrap();
        let up = Vector::new([0.0, 0.0, 1.0]);
        
        Rotor3D::from_vectors(&up, &thrust_dir).unwrap()
    }
}

impl Debug for Quadcopter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quadcopter")
            .field("position", &self.body.state.position)
            .field("velocity", &self.body.state.velocity)
            .field("orientation", &self.body.state.orientation)
            .field("angular_velocity", &self.body.state.angular_velocity)
            .field("motors", &self.motors)
            .finish()
    }
}

pub fn create_test_quadcopter() -> Quadcopter {
    Quadcopter::new(QuadcopterConfig::default())
}
