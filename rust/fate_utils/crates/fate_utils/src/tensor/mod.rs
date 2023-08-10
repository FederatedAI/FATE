mod cb;

use bincode::{deserialize, serialize};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use paillier_tensor::block;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// cipherblock contains ciphertexts and pubkey
///
/// we need `new` method with zero argument (Option::None)
/// for unpickle to work.
#[pyclass(module = "fate_utils.tensor")]
pub struct Cipherblock(Option<block::Cipherblock>);

#[pyclass(module = "fate_utils.tensor")]
pub struct PK {
    pk: Option<fixedpoint::PK>,
}

impl PK {
    fn new(pk: fixedpoint::PK) -> Self {
        Self { pk: Some(pk) }
    }
    fn as_ref(&self) -> &fixedpoint::PK {
        self.pk.as_ref().unwrap()
    }
}

#[pyclass(module = "fate_utils.tensor")]
pub struct SK {
    sk: Option<fixedpoint::SK>,
}

impl SK {
    fn new(sk: fixedpoint::SK) -> Self {
        Self { sk: Some(sk) }
    }
    fn as_ref(&self) -> &fixedpoint::SK {
        self.sk.as_ref().unwrap()
    }
}

#[pyfunction]
fn keygen(bit_size: u32) -> (PK, SK) {
    let (sk, pk) = fixedpoint::keygen(bit_size);
    (PK::new(pk), SK::new(sk))
}

/// public key for paillier system used to encrypt arrays
///
/// Notes: we could not use Generics Types or rule macro here, sad.
#[pymethods]
impl PK {
    #[new]
    fn __new__() -> Self {
        Self { pk: None }
    }
    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(self.as_ref()).unwrap()).to_object(py))
    }
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.pk = Some(deserialize(s.as_bytes()).unwrap());
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    pub fn __richcmp__(&self, other: &PK, cmp: pyo3::basic::CompareOp) -> PyResult<bool> {
        match cmp {
            pyo3::basic::CompareOp::Eq => Ok(self.as_ref() == other.as_ref()),
            _ => Err(PyTypeError::new_err(
                "not supported between instances PK and PK",
            )),
        }
    }
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

/// secret key for paillier system used to encrypt arrays
///
/// Notes: we could not use Generics Types or rule macro here, sad.
#[pymethods]
impl SK {
    #[new]
    fn __new__() -> Self {
        Self { sk: None }
    }
    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(self.as_ref()).unwrap()).to_object(py))
    }
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.sk = Some(deserialize(s.as_bytes()).unwrap());
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    pub fn __richcmp__(&self, other: &SK, cmp: pyo3::basic::CompareOp) -> PyResult<bool> {
        match cmp {
            pyo3::basic::CompareOp::Eq => Ok(self.as_ref() == other.as_ref()),
            _ => Err(PyTypeError::new_err(
                "not supported between instances PK and PK",
            )),
        }
    }
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

/// methods for cipherblock
///
/// Notes: we could not use Generics Types or rule macro here, sad.
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
    pub fn __str__(&self) -> String {
        format!("{:?}", self.0.as_ref().unwrap())
    }
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.0.as_ref().map(|cb| cb.shape.clone()).unwrap()
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

    // rsub
    pub fn rsub_cipherblock(&self, other: &Cipherblock) -> Cipherblock {
        self.rsub_cb(other)
    }
    pub fn rsub_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.rsub_plaintext(other.as_array())
    }
    pub fn rsub_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.rsub_plaintext(other.as_array())
    }
    pub fn rsub_plaintext_i64(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.rsub_plaintext(other.as_array())
    }
    pub fn rsub_plaintext_i32(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.rsub_plaintext(other.as_array())
    }
    pub fn rsub_plaintext_scalar_f64(&self, other: f64) -> Cipherblock {
        self.rsub_plaintext_scalar(other)
    }
    pub fn rsub_plaintext_scalar_f32(&self, other: f32) -> Cipherblock {
        self.rsub_plaintext_scalar(other)
    }
    pub fn rsub_plaintext_scalar_i64(&self, other: i64) -> Cipherblock {
        self.rsub_plaintext_scalar(other)
    }
    pub fn rsub_plaintext_scalar_i32(&self, other: i32) -> Cipherblock {
        self.rsub_plaintext_scalar(other)
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

    pub fn slice0(&self, i: usize) -> Cipherblock {
        Cipherblock(Some(self.0.as_ref().unwrap().slice0(i)))
    }
}

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule_tensor = PyModule::new(py, "tensor")?;
    m.add_submodule(submodule_tensor)?;
    submodule_tensor.add_class::<Cipherblock>()?;
    submodule_tensor.add_class::<PK>()?;
    submodule_tensor.add_class::<SK>()?;
    submodule_tensor.add_function(wrap_pyfunction!(keygen, m)?)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("fate_utils.tensor", submodule_tensor)?;
    Ok(())
}
