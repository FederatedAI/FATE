pub mod block;
pub mod fixedpoint;
mod math;
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
        self.sk.decrypt_array(array).into_pyarray(py)
    }
    #[cfg(feature = "rayon")]
    fn decrypt_f32_par<'py>(&self, a: &Cipherblock, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array(array).into_pyarray(py)
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
    pub fn add_plaintext(&self, other: PyReadonlyArrayDyn<f64> ) -> Cipherblock {
        self._add_plaintext(other)
    }
    pub fn sub_plaintext(&self, other: PyReadonlyArrayDyn<f64> ) -> Cipherblock {
        self._sub_plaintext(other)
    }
    pub fn mul_plaintext(&self, other: PyReadonlyArrayDyn<f64> ) -> Cipherblock {
        self._mul_plaintext(other)
    }
    #[cfg(feature="rayon")]
    pub fn add_cipherblock_par(&self, other: &Cipherblock) -> Cipherblock {
        self._add_cipherblock_par(other)
    }
}
macro_rules! impl_ops_cipher {
    ($name:ident,$fn:ident) => {
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            Cipherblock::binary_cipher(self, other, |lhs, rhs| block::Cipherblock::$fn(lhs, rhs))
        }
    };
    ($name:ident,$fn:ident,$feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            Cipherblock::binary_cipher(self, other, |lhs, rhs| block::Cipherblock::$fn(lhs, rhs))
        }
    };
}
macro_rules! impl_ops_plain {
    ($name:ident,$fn:ident) => {
        pub fn $name(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
            Cipherblock::binary_plain(self, other, |lhs, rhs| block::Cipherblock::$fn(lhs, rhs))
        }
    };
    ($name:ident,$fn:ident,$feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name(&self, other: PyReadonlyArrayDyn<f64>) -> Cipherblock {
            Cipherblock::binary_plain(self, other, |lhs, rhs| block::Cipherblock::$fn(lhs, rhs))
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
    impl_ops_cipher!(_add_cipherblock, add_cipherblock);
    impl_ops_cipher!(_sub_cipherblock, sub_cipherblock);
    impl_ops_plain!(_add_plaintext, add_plaintext);
    impl_ops_plain!(_sub_plaintext, sub_plaintext);
    impl_ops_plain!(_mul_plaintext, mul_plaintext);

    //par
    impl_ops_cipher!(_add_cipherblock_par, add_cipherblock_par, rayon);
    impl_ops_plain!(_add_plaintext_par, add_plaintext_par, rayon);
}
#[pymodule]
fn fate_tensor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Cipherblock>()?;
    m.add_class::<PK>()?;
    m.add_class::<SK>()?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    Ok(())
}
