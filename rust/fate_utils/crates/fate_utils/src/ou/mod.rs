mod ou;
use pyo3::prelude::*;

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "ou")?;
    ou::register(py, submodule)?;
    m.add_submodule(submodule)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_utils.ou", submodule)?;
    Ok(())
}
