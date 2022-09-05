use libsm;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use pyo3::wrap_pyfunction;

/// hash of bytes
#[pyfunction]
fn sm3_hash(py: Python, a: &[u8]) -> PyObject {
    let mut hash = libsm::sm3::hash::Sm3Hash::new(a);
    let digest: [u8; 32] = hash.get_hash();
    PyByteArray::new(py, &digest).into()
}


pub(crate) fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sm3_hash, m)?)?;
    Ok(())
}
