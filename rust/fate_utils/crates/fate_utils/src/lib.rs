extern crate core;

mod hash;
mod histogram;
mod psi;
mod quantile;
mod secure_aggregation_helper;
mod paillier;

use pyo3::prelude::*;

#[pymodule]
fn fate_utils(py: Python, m: &PyModule) -> PyResult<()> {
    quantile::register(py, m)?;
    hash::register(py, m)?;
    psi::register(py, m)?;
    histogram::register(py, m)?;
    paillier::register(py, m)?;
    secure_aggregation_helper::register(py, m)?;
    Ok(())
}
