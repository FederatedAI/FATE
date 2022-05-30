use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::montgomery::MontgomeryPoint;
use curve25519_dalek::scalar::Scalar;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass]
struct Secret(Scalar);

#[pymethods]
impl Secret {
    #[new]
    fn new(bytes: &[u8]) -> Self {
        Self(Scalar::from_bytes_mod_order(bytes.try_into().expect("private key accept 32 bytes")))
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

#[pymodule]
fn psi_ecdh_curve25519(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Secret>()?;
    Ok(())
}
