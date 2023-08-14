use std::collections::HashMap;

use ndarray;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::distributions::Uniform;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::OsRng;
use x25519_dalek::{EphemeralSecret, PublicKey};

#[pyclass]
struct DiffieHellman {
    private_key: Option<EphemeralSecret>,
    public_key: PublicKey,
}

#[pymethods]
impl DiffieHellman {
    #[new]
    fn new() -> Self {
        let private_key = EphemeralSecret::new(OsRng);
        let public_key = PublicKey::from(&private_key);
        Self {
            private_key: Some(private_key),
            public_key,
        }
    }

    fn get_public_key(&self) -> &[u8] {
        return self.public_key.as_bytes();
    }
    pub fn diffie_hellman(&mut self, py: Python, other_public_key: &[u8]) -> PyResult<Py<PyBytes>> {
        let private_key = match self.private_key.take() {
            Some(key) => key,
            None => return Err(PyTypeError::new_err("Private key not found")),
        };

        let other_public_key: [u8; 32] = match other_public_key.try_into() {
            Ok(key) => key,
            Err(_) => {
                return Err(PyTypeError::new_err(
                    "slice with incorrect length, should be 32 bytes",
                ))
            }
        };

        let shared_secret = private_key.diffie_hellman(&PublicKey::from(other_public_key));
        Ok(PyBytes::new(py, shared_secret.as_bytes()).into())
    }
}

#[pyclass]
struct RandomMixState {
    rank: usize,
    random_state: ChaCha20Rng,
    index: usize,
}

#[pyclass]
struct RandomMix {
    rank: usize,
    states: Vec<RandomMixState>,
}

#[pymethods]
impl RandomMix {
    #[new]
    fn new(seeds: HashMap<usize, Vec<u8>>, rank: usize) -> PyResult<Self> {
        let states: Result<Vec<_>, _> = seeds
            .iter()
            .map(|(k, v)| {
                if k == &rank {
                    return Err(PyErr::new::<PyIndexError, _>(
                        "Seed should not contain the rank",
                    ));
                }
                let seed_arr = <&[u8; 32]>::try_from(&v[..])
                    .map_err(|_| PyErr::new::<PyTypeError, _>("Seed should be a 32-byte array"))?;
                let random_state = ChaCha20Rng::from_seed(*seed_arr);
                Ok(RandomMixState {
                    rank: *k,
                    random_state,
                    index: 0,
                })
            })
            .collect();
        match states {
            Ok(states) => Ok(Self { rank, states }),
            Err(e) => Err(e),
        }
    }

    fn mix_one(
        &mut self,
        py: Python,
        input: PyReadonlyArrayDyn<f64>,
        weight: Option<f64>,
    ) -> (Py<PyArrayDyn<f64>>, Py<PyArrayDyn<f64>>) {
        let (mut output_decimal_array, mut output_integer_array) = {
            if let Some(w) = weight {
                let input = input.as_array().map(|x| x * w);
                (input.map(|x| x.fract()), input.map(|x| x.trunc()))
            } else {
                let input = input.as_array();
                (input.map(|x| x.fract()), input.map(|x| x.trunc()))
            }
        };
        let range = Uniform::new(-1e7f64, 1e7f64);
        output_decimal_array
            .iter_mut()
            .zip(output_integer_array.iter_mut())
            .for_each(|(output_decimal, output_integer)| {
                for state in self.states.iter_mut() {
                    let rand = state.random_state.sample(range);
                    state.index += 1;
                    if state.rank < self.rank {
                        *output_decimal += rand.fract();
                        *output_integer += rand.trunc();
                    } else {
                        *output_decimal -= rand.fract();
                        *output_integer -= rand.trunc();
                    }
                }
            });
        (
            output_decimal_array.into_pyarray(py).to_owned(),
            output_integer_array.into_pyarray(py).to_owned(),
        )
    }
    fn mix(
        &mut self,
        py: Python,
        inputs: Vec<PyReadonlyArrayDyn<f64>>,
        weight: Option<f64>,
    ) -> Vec<(Py<PyArrayDyn<f64>>, Py<PyArrayDyn<f64>>)> {
        inputs
            .into_iter()
            .map(|input| self.mix_one(py, input, weight))
            .collect()
    }

    fn get_index(&self, rank: usize) -> PyResult<usize> {
        let state = self
            .states
            .iter()
            .find(|state| state.rank == rank)
            .ok_or(PyErr::new::<PyIndexError, _>(format!(
                "Rank {} not found",
                rank
            )))?;
        Ok(state.index)
    }
}

#[pyclass]
struct MixAggregate {
    decimal_sum: Vec<ArrayD<f64>>,
    integer_sum: Vec<ArrayD<f64>>,
}

#[pymethods]
impl MixAggregate {
    #[new]
    fn new() -> Self {
        Self {
            decimal_sum: Vec::new(),
            integer_sum: Vec::new(),
        }
    }
    fn aggregate(&mut self, inputs: Vec<(PyReadonlyArrayDyn<f64>, PyReadonlyArrayDyn<f64>)>) {
        inputs
            .into_iter()
            .enumerate()
            .for_each(|(i, (decimal, integer))| {
                if i >= self.decimal_sum.len() {
                    self.decimal_sum.push(decimal.as_array().to_owned());
                    self.integer_sum.push(integer.as_array().to_owned());
                } else {
                    self.decimal_sum[i] += &decimal.as_array();
                    self.integer_sum[i] += &integer.as_array();
                }
            });
    }
    fn finalize(&self, py: Python, weight: Option<f64>) -> Vec<Py<PyArrayDyn<f64>>> {
        self.decimal_sum
            .iter()
            .zip(self.integer_sum.iter())
            .map(|(decimal, integer)| {
                let mut output = decimal.clone();
                output += integer;
                if let Some(w) = weight {
                    output /= w;
                }
                output.into_pyarray(py).to_owned()
            })
            .collect()
    }
}

pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule_secure_aggregation_helper = PyModule::new(py, "secure_aggregation_helper")?;
    submodule_secure_aggregation_helper.add_class::<RandomMix>()?;
    submodule_secure_aggregation_helper.add_class::<MixAggregate>()?;
    submodule_secure_aggregation_helper.add_class::<DiffieHellman>()?;
    m.add_submodule(submodule_secure_aggregation_helper)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "fate_utils.secure_aggregation_helper",
        submodule_secure_aggregation_helper,
    )?;
    Ok(())
}
