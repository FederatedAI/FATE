mod paillier;
mod integer_paillier;
use pyo3::prelude::*;


pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "paillier")?;
    paillier::register(py, submodule)?;
    m.add_submodule(submodule)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_utils.paillier", submodule)?;
    Ok(())
}
