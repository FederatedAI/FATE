mod sm3;
use pyo3::prelude::*;

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule_hash = PyModule::new(py, "hash")?;
    sm3::register(py, submodule_hash)?;
    m.add_submodule(submodule_hash)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_crypto.hash", submodule_hash)?;
    Ok(())
}
