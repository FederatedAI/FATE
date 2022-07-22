mod psi;
mod hash;
use pyo3::prelude::*;

#[pymodule]
fn fate_crypto(py: Python, m: &PyModule) -> PyResult<()> {
    psi::register(py, m)?;
    hash::register(py, m)?;
    Ok(())
}
