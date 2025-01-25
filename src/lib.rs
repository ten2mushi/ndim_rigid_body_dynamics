use std::error::Error;
pub mod utils;
// pub mod quadcopter;

use crate::utils::rotors::{Vector, Rotor3D, Bivector};
use crate::utils::quadcopter::{Quadcopter, create_test_quadcopter};
use crate::utils::constants::{GRAVITY, AIR_DENSITY, VISCOSITY};
use pyo3::prelude::*;


#[pyclass]
struct QuadcopterSim {
    quad: Quadcopter,
}

#[pymethods]
impl QuadcopterSim {
    #[new]
    fn new() -> Self {
        Self {
            quad: create_test_quadcopter()
        }
    }

    fn step(
        &mut self,
        control_inputs: Vec<f64>,
        dt: f64
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let controls: [f64; 4] = control_inputs.try_into()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Control inputs must be length 4"
            ))?;

        self.quad.update(dt, &controls);

        Ok((
            self.quad.body.state.position.components.to_vec(),
            self.quad.body.state.velocity.components.to_vec(),
            vec![
                self.quad.body.state.orientation.scalar,
                self.quad.body.state.orientation.bivector.components[0],
                self.quad.body.state.orientation.bivector.components[1],
                self.quad.body.state.orientation.bivector.components[2]
            ],
            self.quad.body.state.angular_velocity.components.to_vec()
        ))
    }
}

/// Python module
#[pymodule]
#[pyo3(name = "rotors_quad_sim")]
fn rotors_quad_sim_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<QuadcopterSim>()?;
    Ok(())
}