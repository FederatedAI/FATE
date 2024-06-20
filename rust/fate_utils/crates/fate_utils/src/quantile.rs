use bincode::{deserialize, serialize};
use ndarray::prelude::*;
use ndarray::{Array, ArrayView1, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyTuple;
use quantile::greenwald_khanna;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug, Serialize, Deserialize)]
struct Ordf64(f64);
impl Eq for Ordf64 {}
impl Ord for Ordf64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
    fn max(self, other: Self) -> Self {
        Ordf64(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        Ordf64(self.0.min(other.0))
    }
}

#[pyclass(module = "fate_utils.quantile")]
pub struct QuantileSummaryStream(Option<greenwald_khanna::Stream<Ordf64>>);

impl QuantileSummaryStream {
    fn new(epsilon: Option<f64>) -> Self {
        match epsilon {
            Some(e) => Self(Some(greenwald_khanna::Stream::new(e))),
            None => Self(None),
        }
    }
}

#[pymethods]
impl QuantileSummaryStream {
    #[new]
    #[pyo3(signature = (*args))]
    fn __new__(args: &PyTuple) -> PyResult<Self> {
        match args.len() {
            0 => Ok(QuantileSummaryStream::new(None)),
            1 => args
                .get_item(0)
                .unwrap()
                .extract::<f64>()
                .map_err(|e| PyTypeError::new_err(e.to_string())) // convert error to pyerr
                .map(|epsion| QuantileSummaryStream::new(Some(epsion))),
            _ => Err(PyTypeError::new_err("accept zero or one positional args")),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.0).unwrap()).to_object(py))
    }
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.0 = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    pub fn insert_array(&mut self, data: PyReadonlyArray1<f64>) -> PyResult<()> {
        for d in data.as_array().into_iter() {
            self.0.as_mut().unwrap().insert(Ordf64(*d))
        }
        Ok(())
    }
    pub fn queries(&self, phi: Vec<f64>) -> Vec<f64> {
        phi.iter()
            .map(|p| self.0.as_ref().unwrap().quantile(*p).0)
            .collect()
    }
    pub fn merge(&self, other: &QuantileSummaryStream) -> QuantileSummaryStream {
        QuantileSummaryStream(Some(
            self.0.as_ref().unwrap().merge(&other.0.as_ref().unwrap()),
        ))
    }
}

#[pyfunction]
fn summary_f64_ix2(data: PyReadonlyArray2<f64>, epsilon: f64) -> Vec<QuantileSummaryStream> {
    let array = data.as_array();
    let mut outputs: Vec<_> = (0..array.shape()[1])
        .map(|_x| QuantileSummaryStream::new(Some(epsilon)))
        .collect();
    for j in 0..array.shape()[1] {
        let arr = array.index_axis(Axis(1), j);
        for d in arr.into_iter() {
            outputs[j].0.as_mut().unwrap().insert(Ordf64(*d));
        }
    }
    outputs
}

fn quantile_f64(data: ArrayView1<f64>, q: &Vec<f64>, epsilon: f64) -> Vec<f64> {
    let mut stream = greenwald_khanna::Stream::new(epsilon);
    for d in data.into_iter() {
        stream.insert(Ordf64(*d))
    }
    println!("size is {}", stream.s());
    q.iter().map(|phi| stream.quantile(*phi).0).collect()
}

#[pyfunction]
fn quantile_f64_ix1(data: PyReadonlyArray1<f64>, q: Vec<f64>, epsilon: f64) -> Vec<f64> {
    quantile_f64(data.as_array(), &q, epsilon)
}

#[pyfunction]
fn quantile_f64_ix2<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    q: Vec<f64>,
    epsilon: f64,
) -> &'py PyArray2<f64> {
    let array = data.as_array();
    let mut a = Array::<f64, _>::zeros((q.len(), array.shape()[1]).f());
    for (j, mut col) in a.axis_iter_mut(Axis(1)).enumerate() {
        let arr = array.index_axis(Axis(1), j);
        let quantile_sub = quantile_f64(arr, &q, epsilon);
        for (i, row) in col.iter_mut().enumerate() {
            *row = quantile_sub[i];
        }
    }
    a.into_pyarray(py)
}

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule_quantile = PyModule::new(py, "quantile")?;
    submodule_quantile.add_class::<QuantileSummaryStream>()?;
    m.add_submodule(submodule_quantile)?;
    submodule_quantile.add_function(wrap_pyfunction!(quantile_f64_ix1, submodule_quantile)?)?;
    submodule_quantile.add_function(wrap_pyfunction!(quantile_f64_ix2, submodule_quantile)?)?;
    submodule_quantile.add_function(wrap_pyfunction!(summary_f64_ix2, submodule_quantile)?)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_utils.quantile", submodule_quantile)?;
    Ok(())
}
