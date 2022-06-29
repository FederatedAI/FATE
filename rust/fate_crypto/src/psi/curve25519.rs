use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::montgomery::MontgomeryPoint;
use curve25519_dalek::scalar::Scalar;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::ToPyObject;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

#[pyclass(module = "fate_crypto.psi", name = "Curve25519")]
struct Secret(Scalar);

#[pymethods]
impl Secret {
    #[new]
    #[args(args = "*")]
    fn new(args: &PyTuple) -> PyResult<Self> {
        if args.len() > 1 {
            Err(PyTypeError::new_err("accept zero or one positional args"))
        } else {
            let byte32: [u8; 32] = match args.len() {
                0 => {
                    let mut bytes: [u8; 32] = [0; 32];
                    StdRng::from_entropy().fill_bytes(&mut bytes);
                    bytes
                }
                1 => {
                    let bytes: [u8; 32] = args.get_item(0).unwrap().extract::<[u8; 32]>().unwrap();
                    bytes
                }
                _ => unreachable!(),
            };
            Ok(Self(Scalar::from_bytes_mod_order(byte32)))
        }
    }
    pub fn get_private_key(&self, py: Python) -> PyObject {
        PyBytes::new(py, self.0.as_bytes()).to_object(py)
    }
    pub fn __getstate__(&self, py: Python) -> PyObject {
        PyBytes::new(py, self.0.as_bytes()).to_object(py)
    }
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.0 =
                    Scalar::from_bytes_mod_order(s.as_bytes().try_into().expect("invalid state"));
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    #[pyo3(text_signature = "($self, bytes)")]
    fn encrypt(&self, bytes: &[u8], py: Python) -> PyObject {
        PyBytes::new(
            py,
            (EdwardsPoint::hash_from_bytes::<sha2::Sha512>(bytes).to_montgomery() * self.0)
                .as_bytes(),
        )
        .into()
    }
    #[pyo3(text_signature = "($self, their_public)")]
    fn diffie_hellman(&self, their_public: &[u8], py: Python) -> PyObject {
        PyBytes::new(
            py,
            (MontgomeryPoint(
                their_public
                    .try_into()
                    .expect("diffie_hellman accpet 32 bytes pubkey"),
            ) * self.0)
                .as_bytes(),
        )
        .into()
    }
}

pub(crate) fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Secret>()?;
    Ok(())
}
