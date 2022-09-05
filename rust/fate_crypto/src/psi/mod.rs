mod curve25519;
use pyo3::prelude::*;

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule_psi = PyModule::new(py, "psi")?;
    curve25519::register(py, submodule_psi)?;
    m.add_submodule(submodule_psi)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_crypto.psi", submodule_psi)?;
    Ok(())
}
