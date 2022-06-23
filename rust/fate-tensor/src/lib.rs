pub mod block;
pub mod fixedpoint;
pub mod math;
pub mod paillier;

use bincode::{deserialize, serialize};
use ndarray::ArrayViewD;
use numpy::convert::IntoPyArray;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

#[pymethods]
impl PK {
    fn encrypt_f64(&self, a: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        Cipherblock(Some(self.pk.encrypt_array(a.as_array())))
    }
    fn encrypt_f32(&self, a: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        Cipherblock(Some(self.pk.encrypt_array(a.as_array())))
    }
    #[cfg(feature = "rayon")]
    fn encrypt_f64_par(&self, a: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        Cipherblock(Some(self.pk.encrypt_array_par(a.as_array())))
    }
    #[cfg(feature = "rayon")]
    fn encrypt_f32_par(&self, a: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        Cipherblock(Some(self.pk.encrypt_array_par(a.as_array())))
    }
}

#[pymethods]
impl SK {
    fn decrypt_f64<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f64> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array(array).into_pyarray(py)
    }
    fn decrypt_f32<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array(array).into_pyarray(py)
    }
    #[cfg(feature = "rayon")]
    fn decrypt_f64_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f64> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array_par(array).into_pyarray(py)
    }
    #[cfg(feature = "rayon")]
    fn decrypt_f32_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array_par(array).into_pyarray(py)
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
    pub fn add_cipherblock(&self, other: &Cipherblock) -> Cipherblock {
        self._add_cipherblock(other)
    }
    pub fn sub_cipherblock(&self, other: &Cipherblock) -> Cipherblock {
        self._sub_cipherblock(other)
    }
    pub fn add_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self._add_plaintext_f64(other)
    }
    pub fn sub_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self._sub_plaintext_f64(other)
    }
    pub fn mul_plaintext_f64(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self._mul_plaintext_f64(other)
    }
    pub fn add_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self._add_plaintext_f32(other)
    }
    pub fn sub_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self._sub_plaintext_f32(other)
    }
    pub fn mul_plaintext_f32(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self._mul_plaintext_f32(other)
    }
    #[cfg(feature = "rayon")]
    pub fn add_cipherblock_par(&self, other: &Cipherblock) -> Cipherblock {
        self._add_cipherblock(other)
    }
    #[cfg(feature = "rayon")]
    pub fn sub_cipherblock_par(&self, other: &Cipherblock) -> Cipherblock {
        self._sub_cipherblock(other)
    }
    #[cfg(feature = "rayon")]
    pub fn add_plaintext_f64_par(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self._add_plaintext_f64(other)
    }
    #[cfg(feature = "rayon")]
    pub fn sub_plaintext_f64_par(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self._sub_plaintext_f64(other)
    }
    #[cfg(feature = "rayon")]
    pub fn mul_plaintext_f64_par(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
        self._mul_plaintext_f64(other)
    }
    #[cfg(feature = "rayon")]
    pub fn add_plaintext_f32_par(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self._add_plaintext_f32_par(other)
    }
    #[cfg(feature = "rayon")]
    pub fn sub_plaintext_f32_par(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self._sub_plaintext_f32_par(other)
    }
    #[cfg(feature = "rayon")]
    pub fn mul_plaintext_f32_par(&self, other: PyReadonlyArrayDyn<f32>) -> Cipherblock {
        self._mul_plaintext_f32_par(other)
    }
}
macro_rules! impl_ops_cipher {
    ($name:ident,$fn:expr) => {
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            Cipherblock::binary_cipher(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_cb(lhs, rhs, $fn)
            })
        }
    };
    ($name:ident,$fn:expr,$feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            Cipherblock::binary_cipher(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_cb_par(lhs, rhs, $fn)
            })
        }
    };
}
macro_rules! impl_ops_plain {
    ($name:ident,$fn:expr,$T:ty) => {
        pub fn $name(&self, other: PyReadonlyArrayDyn<$T>) -> Cipherblock {
            Cipherblock::binary_plain(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_pt(lhs, rhs, $fn)
            })
        }
    };
    ($name:ident,$fn:expr,$T:ty,$feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name(&self, other: PyReadonlyArrayDyn<$T>) -> Cipherblock {
            Cipherblock::binary_plain(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_pt_par(lhs, rhs, $fn)
            })
        }
    };
}
impl Cipherblock {
    fn binary_plain<F, T>(&self, other: PyReadonlyArrayDyn<T>, func: F) -> Cipherblock
    where
        F: Fn(&block::Cipherblock, ArrayViewD<T>) -> block::Cipherblock,
        T: numpy::Element,
    {
        let a = self.get_cb();
        let b = other.as_array();
        Cipherblock(Some(func(a, b)))
    }

    fn binary_cipher<F>(&self, other: &Cipherblock, func: F) -> Cipherblock
    where
        F: Fn(&block::Cipherblock, &block::Cipherblock) -> block::Cipherblock,
    {
        let a = self.get_cb();
        let b = other.get_cb();
        Cipherblock(Some(func(a, b)))
    }
    fn get_cb(&self) -> &block::Cipherblock {
        self.0.as_ref().unwrap()
    }
    impl_ops_cipher!(_add_cipherblock, fixedpoint::CT::add);
    impl_ops_cipher!(_sub_cipherblock, fixedpoint::CT::sub);
    impl_ops_plain!(_add_plaintext_f64, fixedpoint::CT::add_pt, f64);
    impl_ops_plain!(_sub_plaintext_f64, fixedpoint::CT::sub_pt, f64);
    impl_ops_plain!(_mul_plaintext_f64, fixedpoint::CT::mul, f64);
    impl_ops_plain!(_add_plaintext_f32, fixedpoint::CT::add_pt, f32);
    impl_ops_plain!(_sub_plaintext_f32, fixedpoint::CT::sub_pt, f32);
    impl_ops_plain!(_mul_plaintext_f32, fixedpoint::CT::mul, f32);

    //par
    impl_ops_cipher!(_add_cipherblock_par, fixedpoint::CT::add, rayon);
    impl_ops_cipher!(_sub_cipherblock_par, fixedpoint::CT::sub, rayon);
    impl_ops_plain!(_add_plaintext_f64_par, fixedpoint::CT::add_pt, f64, rayon);
    impl_ops_plain!(_sub_plaintext_f64_par, fixedpoint::CT::sub_pt, f64, rayon);
    impl_ops_plain!(_mul_plaintext_f64_par, fixedpoint::CT::mul, f64, rayon);
    impl_ops_plain!(_add_plaintext_f32_par, fixedpoint::CT::add_pt, f32, rayon);
    impl_ops_plain!(_sub_plaintext_f32_par, fixedpoint::CT::sub_pt, f32, rayon);
    impl_ops_plain!(_mul_plaintext_f32_par, fixedpoint::CT::mul, f32, rayon);
}
#[pymodule]
fn fate_tensor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Cipherblock>()?;
    m.add_class::<PK>()?;
    m.add_class::<SK>()?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    Ok(())
}
