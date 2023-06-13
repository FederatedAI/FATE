mod hash;
mod psi;
mod quantile;
mod tensor;

use pyo3::prelude::*;

#[pymodule]
fn fate_utils(py: Python, m: &PyModule) -> PyResult<()> {
    tensor::register(py, m)?;
    quantile::register(py, m)?;
    hash::register(py, m)?;
    psi::register(py, m)?;
    Ok(())
}
