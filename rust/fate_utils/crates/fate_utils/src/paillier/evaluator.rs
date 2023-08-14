use fixedpoint::CT;
<<<<<<< HEAD
use ndarray::prelude::*;
=======
>>>>>>> dev-2.0.0-beta
use pyo3::prelude::*;
use super::paillier;

#[pyclass]
pub struct Evaluator {}

#[pymethods]
impl Evaluator {
    #[staticmethod]
    fn cat(vec_list: Vec<PyRef<paillier::FixedpointPaillierVector>>) -> PyResult<paillier::FixedpointPaillierVector> {
        let mut data = vec![CT::zero(); 0];
        for vec in vec_list {
            data.extend(vec.data.clone());
        }
        Ok(paillier::FixedpointPaillierVector { data })
    }
    #[staticmethod]
    fn slice_indexes(a: &paillier::FixedpointPaillierVector, indexes: Vec<usize>) -> PyResult<paillier::FixedpointPaillierVector> {
        let data = indexes
            .iter()
            .map(|i| a.data[*i].clone())
            .collect::<Vec<_>>();
        Ok(paillier::FixedpointPaillierVector { data })
    }
}

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Evaluator>()?;
    Ok(())
}
