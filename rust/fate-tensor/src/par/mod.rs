use bincode::{deserialize, serialize};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use crate::fixedpoint;
use crate::block;

mod cb;

#[pyclass(module = "fate_tensor.par")]
pub struct Cipherblock(Option<block::Cipherblock>);

#[pyclass(module = "fate_tensor.par")]
pub struct PK {
    pk: fixedpoint::PK,
}

#[pyclass(module = "fate_tensor.par")]
pub struct SK {
    sk: fixedpoint::SK,
}

#[pyfunction]
fn keygen(bit_size: u32) -> (PK, SK) {
    let (sk, pk) = fixedpoint::keygen(bit_size);
    (PK { pk }, SK { sk })
}

#[pyfunction]
fn set_num_threads(num_threads: usize) {
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
}

#[pymethods]
impl PK {
    fn encrypt_f64(&self, a: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.encrypt_array(a.as_array())
    }
    fn encrypt_f32(&self, a: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.encrypt_array(a.as_array())
    }
    fn encrypt_i64(&self, a: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.encrypt_array(a.as_array())
    }
    fn encrypt_i32(&self, a: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.encrypt_array(a.as_array())
    }
}

#[pymethods]
impl SK {
    fn decrypt_f64<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f64> {
        self.decrypt_array(a).into_pyarray(py)
    }
    fn decrypt_f32<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        self.decrypt_array(a).into_pyarray(py)
    }
    fn decrypt_i64<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<i64> {
        self.decrypt_array(a).into_pyarray(py)
    }
    fn decrypt_i32<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<i32> {
        self.decrypt_array(a).into_pyarray(py)
    }
}

#[pymethods]
impl Cipherblock {
    #[new]
    fn __new__() -> Self {
        Cipherblock(None)
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
    // add
    pub fn add_cipherblock(&self, other: &Cipherblock) -> Cipherblock {
        self.add_cb(other)
    }
    pub fn add_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.add_plaintext(other.as_array())
    }
    pub fn add_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.add_plaintext(other.as_array())
    }
    pub fn add_plaintext_i64(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.add_plaintext(other.as_array())
    }
    pub fn add_plaintext_i32(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.add_plaintext(other.as_array())
    }
    pub fn add_plaintext_scalar_f64(&self, other: f64) -> Cipherblock {
        self.add_plaintext_scalar(other)
    }
    pub fn add_plaintext_scalar_f32(&self, other: f32) -> Cipherblock {
        self.add_plaintext_scalar(other)
    }
    pub fn add_plaintext_scalar_i64(&self, other: i64) -> Cipherblock {
        self.add_plaintext_scalar(other)
    }
    pub fn add_plaintext_scalar_i32(&self, other: i32) -> Cipherblock {
        self.add_plaintext_scalar(other)
    }

    // sub
    pub fn sub_cipherblock(&self, other: &Cipherblock) -> Cipherblock {
        self.sub_cb(other)
    }
    pub fn sub_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.sub_plaintext(other.as_array())
    }
    pub fn sub_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.sub_plaintext(other.as_array())
    }
    pub fn sub_plaintext_i64(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.sub_plaintext(other.as_array())
    }
    pub fn sub_plaintext_i32(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.sub_plaintext(other.as_array())
    }
    pub fn sub_plaintext_scalar_f64(&self, other: f64) -> Cipherblock {
        self.sub_plaintext_scalar(other)
    }
    pub fn sub_plaintext_scalar_f32(&self, other: f32) -> Cipherblock {
        self.sub_plaintext_scalar(other)
    }
    pub fn sub_plaintext_scalar_i64(&self, other: i64) -> Cipherblock {
        self.sub_plaintext_scalar(other)
    }
    pub fn sub_plaintext_scalar_i32(&self, other: i32) -> Cipherblock {
        self.sub_plaintext_scalar(other)
    }

    // mul
    pub fn mul_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.mul_plaintext(other.as_array())
    }
    pub fn mul_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.mul_plaintext(other.as_array())
    }
    pub fn mul_plaintext_i64(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.mul_plaintext(other.as_array())
    }
    pub fn mul_plaintext_i32(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.mul_plaintext(other.as_array())
    }
    pub fn mul_plaintext_scalar_f64(&self, other: f64) -> Cipherblock {
        self.mul_plaintext_scalar(other)
    }
    pub fn mul_plaintext_scalar_f32(&self, other: f32) -> Cipherblock {
        self.mul_plaintext_scalar(other)
    }
    pub fn mul_plaintext_scalar_i64(&self, other: i64) -> Cipherblock {
        self.mul_plaintext_scalar(other)
    }
    pub fn mul_plaintext_scalar_i32(&self, other: i32) -> Cipherblock {
        self.mul_plaintext_scalar(other)
    }

    // matmul
    pub fn matmul_plaintext_ix2_f64(&self, other: PyReadonlyArray2<f64>) -> Cipherblock {
        self.matmul_plaintext_ix2(other.as_array())
    }
    pub fn matmul_plaintext_ix2_f32(&self, other: PyReadonlyArray2<f32>) -> Cipherblock {
        self.matmul_plaintext_ix2(other.as_array())
    }
    pub fn matmul_plaintext_ix2_i64(&self, other: PyReadonlyArray2<i64>) -> Cipherblock {
        self.matmul_plaintext_ix2(other.as_array())
    }
    pub fn matmul_plaintext_ix2_i32(&self, other: PyReadonlyArray2<i32>) -> Cipherblock {
        self.matmul_plaintext_ix2(other.as_array())
    }
    pub fn rmatmul_plaintext_ix2_f64(&self, other: PyReadonlyArray2<f64>) -> Cipherblock {
        self.rmatmul_plaintext_ix2(other.as_array())
    }
    pub fn rmatmul_plaintext_ix2_f32(&self, other: PyReadonlyArray2<f32>) -> Cipherblock {
        self.rmatmul_plaintext_ix2(other.as_array())
    }
    pub fn rmatmul_plaintext_ix2_i64(&self, other: PyReadonlyArray2<i64>) -> Cipherblock {
        self.rmatmul_plaintext_ix2(other.as_array())
    }
    pub fn rmatmul_plaintext_ix2_i32(&self, other: PyReadonlyArray2<i32>) -> Cipherblock {
        self.rmatmul_plaintext_ix2(other.as_array())
    }
    pub fn matmul_plaintext_ix1_f64(&self, other: PyReadonlyArray1<f64>) -> Cipherblock {
        self.matmul_plaintext_ix1(other.as_array())
    }
    pub fn matmul_plaintext_ix1_f32(&self, other: PyReadonlyArray1<f32>) -> Cipherblock {
        self.matmul_plaintext_ix1(other.as_array())
    }
    pub fn matmul_plaintext_ix1_i64(&self, other: PyReadonlyArray1<i64>) -> Cipherblock {
        self.matmul_plaintext_ix1(other.as_array())
    }
    pub fn matmul_plaintext_ix1_i32(&self, other: PyReadonlyArray1<i32>) -> Cipherblock {
        self.matmul_plaintext_ix1(other.as_array())
    }
    pub fn rmatmul_plaintext_ix1_f64(&self, other: PyReadonlyArray1<f64>) -> Cipherblock {
        self.rmatmul_plaintext_ix1(other.as_array())
    }
    pub fn rmatmul_plaintext_ix1_f32(&self, other: PyReadonlyArray1<f32>) -> Cipherblock {
        self.rmatmul_plaintext_ix1(other.as_array())
    }
    pub fn rmatmul_plaintext_ix1_i64(&self, other: PyReadonlyArray1<i64>) -> Cipherblock {
        self.rmatmul_plaintext_ix1(other.as_array())
    }
    pub fn rmatmul_plaintext_ix1_i32(&self, other: PyReadonlyArray1<i32>) -> Cipherblock {
        self.rmatmul_plaintext_ix1(other.as_array())
    }
    // agg
    pub fn sum(&self) -> Cipherblock {
        self.sum_cb()
    }
    pub fn mean(&self) -> Cipherblock {
        self.sum_cb()
    }
}

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule_par = PyModule::new(py, "par")?;
    submodule_par.add_function(wrap_pyfunction!(keygen, submodule_par)?)?;
    submodule_par.add_function(wrap_pyfunction!(set_num_threads, submodule_par)?)?;
    m.add_submodule(submodule_par)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_tensor.par", submodule_par)?;
    Ok(())
}
