extern crate core;

mod hash;
mod histogram;
mod psi;
mod quantile;
mod secure_aggregation_helper;
mod tensor;

use pyo3::prelude::*;

#[pymodule]
fn fate_utils(py: Python, m: &PyModule) -> PyResult<()> {
    tensor::register(py, m)?;
    quantile::register(py, m)?;
    hash::register(py, m)?;
    psi::register(py, m)?;
    histogram::register(py, m)?;
    secure_aggregation_helper::register(py, m)?;
    Ok(())
}
