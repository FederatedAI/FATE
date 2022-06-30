pub mod block;
pub mod cb;
pub mod fixedpoint;
pub mod math;
pub mod paillier;

use bincode::{deserialize, serialize};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// cipherblock contains ciphertexts and pubkey
///
/// we need `new` method with zero argument (Option::None)
/// for unpickle to work.
#[pyclass(module = "fate_tensor")]
pub struct Cipherblock(Option<block::Cipherblock>);

#[pyclass(module = "fate_tensor")]
pub struct PK {
    pk: fixedpoint::PK,
}

#[pyclass(module = "fate_tensor")]
pub struct SK {
    sk: fixedpoint::SK,
}

#[pyfunction]
fn keygen(bit_size: u32) -> (PK, SK) {
    let (sk, pk) = fixedpoint::keygen(bit_size);
    (PK { pk }, SK { sk })
}

/// public key for paillier system used to encrypt arrays
///
/// Notes: we could not use Generics Types or rule macro here, sad.
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
    #[cfg(feature = "rayon")]
    fn encrypt_f64_par(&self, a: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.encrypt_array_par(a.as_array())
    }
    #[cfg(feature = "rayon")]
    fn encrypt_f32_par(&self, a: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.encrypt_array_par(a.as_array())
    }
    #[cfg(feature = "rayon")]
    fn encrypt_i64_par(&self, a: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.encrypt_array_par(a.as_array())
    }
    #[cfg(feature = "rayon")]
    fn encrypt_i32_par(&self, a: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.encrypt_array_par(a.as_array())
    }
}

/// secret key for paillier system used to encrypt arrays
///
/// Notes: we could not use Generics Types or rule macro here, sad.
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
    #[cfg(feature = "rayon")]
    fn decrypt_f64_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f64> {
        self.decrypt_array_par(a).into_pyarray(py)
    }
    #[cfg(feature = "rayon")]
    fn decrypt_f32_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        self.decrypt_array_par(a).into_pyarray(py)
    }
    #[cfg(feature = "rayon")]
    fn decrypt_i64_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<i64> {
        self.decrypt_array_par(a).into_pyarray(py)
    }
    #[cfg(feature = "rayon")]
    fn decrypt_i32_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<i32> {
        self.decrypt_array_par(a).into_pyarray(py)
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

    // rayon

    // add
    #[cfg(feature = "rayon")]
    pub fn add_cipherblock_par(&self, other: &Cipherblock) -> Cipherblock {
        self.add_cb_par(other)
    }
    #[cfg(feature = "rayon")]
    pub fn add_plaintext_f64_par(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.add_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn add_plaintext_f32_par(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.add_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn add_plaintext_i64_par(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.add_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn add_plaintext_i32_par(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.add_plaintext_par(other.as_array())
    }

    // sub
    #[cfg(feature = "rayon")]
    pub fn sub_cipherblock_par(&self, other: &Cipherblock) -> Cipherblock {
        self.sub_cb_par(other)
    }
    #[cfg(feature = "rayon")]
    pub fn sub_plaintext_f64_par(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.sub_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn sub_plaintext_f32_par(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.sub_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn sub_plaintext_i64_par(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.sub_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn sub_plaintext_i32_par(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.sub_plaintext_par(other.as_array())
    }

    // mul
    #[cfg(feature = "rayon")]
    pub fn mul_plaintext_f64_par(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self.mul_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn mul_plaintext_f32_par(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self.mul_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn mul_plaintext_i64_par(&self, other: PyReadonlyArrayDyn<i64>) -> Cipherblock {
        self.mul_plaintext_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn mul_plaintext_i32_par(&self, other: PyReadonlyArrayDyn<i32>) -> Cipherblock {
        self.mul_plaintext_par(other.as_array())
    }

    // matmul
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix2_f64_par(&self, other: PyReadonlyArray2<f64>) -> Cipherblock {
        self.matmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix2_f32_par(&self, other: PyReadonlyArray2<f32>) -> Cipherblock {
        self.matmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix2_i64_par(&self, other: PyReadonlyArray2<i64>) -> Cipherblock {
        self.matmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix2_i32_par(&self, other: PyReadonlyArray2<i32>) -> Cipherblock {
        self.matmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix2_f64_par(&self, other: PyReadonlyArray2<f64>) -> Cipherblock {
        self.rmatmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix2_f32_par(&self, other: PyReadonlyArray2<f32>) -> Cipherblock {
        self.rmatmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix2_i64_par(&self, other: PyReadonlyArray2<i64>) -> Cipherblock {
        self.rmatmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix2_i32_par(&self, other: PyReadonlyArray2<i32>) -> Cipherblock {
        self.rmatmul_plaintext_ix2_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix1_f64_par(&self, other: PyReadonlyArray1<f64>) -> Cipherblock {
        self.matmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix1_f32_par(&self, other: PyReadonlyArray1<f32>) -> Cipherblock {
        self.matmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix1_i64_par(&self, other: PyReadonlyArray1<i64>) -> Cipherblock {
        self.matmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix1_i32_par(&self, other: PyReadonlyArray1<i32>) -> Cipherblock {
        self.matmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix1_f64_par(&self, other: PyReadonlyArray1<f64>) -> Cipherblock {
        self.rmatmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix1_f32_par(&self, other: PyReadonlyArray1<f32>) -> Cipherblock {
        self.rmatmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix1_i64_par(&self, other: PyReadonlyArray1<i64>) -> Cipherblock {
        self.rmatmul_plaintext_ix1_par(other.as_array())
    }
    #[cfg(feature = "rayon")]
    pub fn rmatmul_plaintext_ix1_i32_par(&self, other: PyReadonlyArray1<i32>) -> Cipherblock {
        self.rmatmul_plaintext_ix1_par(other.as_array())
    }
}
#[pymodule]
fn fate_tensor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Cipherblock>()?;
    m.add_class::<PK>()?;
    m.add_class::<SK>()?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    Ok(())
}
