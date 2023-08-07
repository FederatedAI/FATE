use fixedpoint::CT;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct PK {
    pub pk: fixedpoint::PK,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct SK {
    pub sk: fixedpoint::SK,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct Coders {
    pub coder: fixedpoint::FixedpointCoder,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct PyCT {
    pub ct: CT,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct FixedpointPaillierVector {
    pub data: Vec<CT>,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct FixedpointPaillierCiphertext {
    pub data: CT,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct FixedpointEncoded {
    pub data: fixedpoint::PT,
}

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct FixedpointVector {
    pub data: Vec<fixedpoint::PT>,
}

#[pymethods]
impl PK {
    fn encrypt_encoded(
        &self,
        fixedpoint: &FixedpointVector,
        obfuscate: bool,
    ) -> FixedpointPaillierVector {
        let data = fixedpoint
            .data
            .iter()
            .map(|x| self.pk.encrypt(x, obfuscate))
            .collect();
        FixedpointPaillierVector { data }
    }
    fn encrypt_encoded_scalar(&self, fixedpoint: &FixedpointEncoded, obfuscate: bool) -> PyCT {
        PyCT {
            ct: self.pk.encrypt(&fixedpoint.data, obfuscate),
        }
    }

    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(PK::default())
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.pk).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.pk = bincode::deserialize(&state).unwrap();
        Ok(())
    }
}

#[pymethods]
impl SK {
    fn decrypt_to_encoded(&self, data: &FixedpointPaillierVector) -> FixedpointVector {
        let data = data.data.iter().map(|x| self.sk.decrypt(x)).collect();
        FixedpointVector { data }
    }
    fn decrypt_to_encoded_scalar(&self, data: &PyCT) -> FixedpointEncoded {
        FixedpointEncoded {
            data: self.sk.decrypt(&data.ct),
        }
    }

    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(SK::default())
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.sk).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.sk = bincode::deserialize(&state).unwrap();
        Ok(())
    }
}

#[pymethods]
impl Coders {
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(Coders::default())
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.coder).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.coder = bincode::deserialize(&state).unwrap();
        Ok(())
    }
    fn encode_f64(&self, data: f64) -> FixedpointEncoded {
        FixedpointEncoded {
            data: self.coder.encode_f64(data),
        }
    }

    fn encode_f64_vec(&self, data: PyReadonlyArray1<f64>) -> FixedpointVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.coder.encode_f64(*x))
            .collect();
        FixedpointVector { data }
    }
    fn decode_f64(&self, data: &FixedpointEncoded) -> f64 {
        self.coder.decode_f64(&data.data)
    }
    fn decode_f64_vec<'py>(&self, data: &FixedpointVector, py: Python<'py>) -> &'py PyArray1<f64> {
        Array1::from(
            data.data
                .iter()
                .map(|x| self.coder.decode_f64(x))
                .collect::<Vec<f64>>(),
        )
            .into_pyarray(py)
    }
    fn encode_f32(&self, data: f32) -> FixedpointEncoded {
        FixedpointEncoded {
            data: self.coder.encode_f32(data),
        }
    }
    fn encode_f32_vec(&self, data: PyReadonlyArray1<f32>) -> FixedpointVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.coder.encode_f32(*x))
            .collect();
        FixedpointVector { data }
    }
    fn decode_f32(&self, data: &FixedpointEncoded) -> f32 {
        self.coder.decode_f32(&data.data)
    }
    fn decode_f32_vec<'py>(&self, data: &FixedpointVector, py: Python<'py>) -> &'py PyArray1<f32> {
        Array1::from(
            data.data
                .iter()
                .map(|x| self.coder.decode_f32(x))
                .collect::<Vec<f32>>(),
        )
            .into_pyarray(py)
    }
    fn encode_i64(&self, data: i64) -> FixedpointEncoded {
        FixedpointEncoded {
            data: self.coder.encode_i64(data),
        }
    }
    fn encode_i64_vec(&self, data: PyReadonlyArray1<i64>) -> FixedpointVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.coder.encode_i64(*x))
            .collect();
        FixedpointVector { data }
    }
    fn decode_i64(&self, data: &FixedpointEncoded) -> i64 {
        self.coder.decode_i64(&data.data)
    }
    fn decode_i64_vec(&self, data: &FixedpointVector) -> Vec<i64> {
        data.data.iter().map(|x| self.coder.decode_i64(x)).collect()
    }
    fn encode_i32(&self, data: i32) -> FixedpointEncoded {
        FixedpointEncoded {
            data: self.coder.encode_i32(data),
        }
    }
    fn encode_i32_vec(&self, data: PyReadonlyArray1<i32>) -> FixedpointVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.coder.encode_i32(*x))
            .collect();
        FixedpointVector { data }
    }
    fn decode_i32(&self, data: &FixedpointEncoded) -> i32 {
        self.coder.decode_i32(&data.data)
    }
    fn decode_i32_vec(&self, data: &FixedpointVector) -> Vec<i32> {
        data.data.iter().map(|x| self.coder.decode_i32(x)).collect()
    }
}

#[pyfunction]
fn keygen(bit_length: u32) -> (SK, PK, Coders) {
    let (sk, pk) = fixedpoint::keygen(bit_length);
    let coder = pk.coder.clone();
    (SK { sk }, PK { pk }, Coders { coder })
}

impl FixedpointPaillierVector {
    #[inline]
    fn iadd_i_j(&mut self, pk: &PK, i: usize, j: usize, size: usize) {
        let mut placeholder = CT::zero();
        for k in 0..size {
            placeholder = std::mem::replace(&mut self.data[i + k], placeholder);
            placeholder.add_assign(&self.data[j + k], &pk.pk);
            placeholder = std::mem::replace(&mut self.data[i + k], placeholder);
        }
    }
}

#[pymethods]
impl FixedpointPaillierVector {
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(FixedpointPaillierVector { data: vec![] })
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.data).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.data = bincode::deserialize(&state).unwrap();
        Ok(())
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.data)
    }

    #[staticmethod]
    fn zeros(size: usize) -> PyResult<Self> {
        let data = vec![CT::zero(); size];
        Ok(FixedpointPaillierVector { data })
    }

    fn slice(&mut self, start: usize, size: usize) -> FixedpointPaillierVector {
        let data = self.data[start..start + size].to_vec();
        FixedpointPaillierVector { data }
    }

    fn slice_indexes(&mut self, indexes: Vec<usize>) -> PyResult<Self> {
        let data = indexes
            .iter()
            .map(|i| self.data[*i].clone())
            .collect::<Vec<_>>();
        Ok(FixedpointPaillierVector { data })
    }
    fn cat(&self, others: Vec<PyRef<FixedpointPaillierVector>>) -> PyResult<Self> {
        let mut data = self.data.clone();
        for other in others {
            data.extend(other.data.clone());
        }
        Ok(FixedpointPaillierVector { data })
    }
    fn i_shuffle(&mut self, indexes: Vec<usize>) {
        let mut visited = vec![false; self.data.len()];
        for i in 0..self.data.len() {
            if visited[i] || indexes[i] == i {
                continue;
            }

            let mut current = i;
            let mut next = indexes[current];
            while !visited[next] && next != i {
                self.data.swap(current, next);
                visited[current] = true;
                current = next;
                next = indexes[current];
            }
            visited[current] = true;
        }
    }
    fn intervals_slice(&mut self, intervals: Vec<(usize, usize)>) -> PyResult<Self> {
        let mut data = vec![];
        for (start, end) in intervals {
            if end > self.data.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "end index out of range: start={}, end={}, data_size={}",
                    start,
                    end,
                    self.data.len()
                )));
            }
            data.extend_from_slice(&self.data[start..end]);
        }
        Ok(FixedpointPaillierVector { data })
    }
    fn iadd_slice(&mut self, pk: &PK, position: usize, other: Vec<PyRef<PyCT>>) {
        for (i, x) in other.iter().enumerate() {
            self.data[position + i] = self.data[position + i].add(&x.ct, &pk.pk);
        }
    }
    fn iadd_vec_self(
        &mut self,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> PyResult<()> {
        if sa == sb {
            if let Some(s) = size {
                if sa + s > self.data.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "end index out of range: sa={}, s={}, data_size={}",
                        sa,
                        s,
                        self.data.len()
                    )));
                }
                self.data[sa..sa + s]
                    .iter_mut()
                    .for_each(|x| x.i_double(&pk.pk));
            } else {
                self.data[sa..].iter_mut().for_each(|x| x.i_double(&pk.pk));
            }
        } else if sa < sb {
            // it's safe to update from left to right
            if let Some(s) = size {
                if sb + s > self.data.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "end index out of range: sb={}, s={}, data_size={}",
                        sb,
                        s,
                        self.data.len()
                    )));
                }
                self.iadd_i_j(&pk, sb, sa, s);
            } else {
                self.iadd_i_j(&pk, sb, sa, self.data.len() - sb);
            }
        } else {
            // it's safe to update from right to left
            if let Some(s) = size {
                if sa + s > self.data.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "end index out of range: sa={}, s={}, data_size={}",
                        sa,
                        s,
                        self.data.len()
                    )));
                }
                self.iadd_i_j(&pk, sa, sb, s);
            } else {
                self.iadd_i_j(&pk, sa, sb, self.data.len() - sa);
            }
        }

        // match size {
        //     Some(s) => {
        //         let ea = sa + s;
        //         let eb = sb + s;
        //         if ea > self.data.len() {
        //             return Err(PyRuntimeError::new_err(format!("end index out of range: sa={}, ea={}, data_size={}", sa, ea, self.data.len())));
        //         }
        //         if eb > self.data.len() {
        //             return Err(PyRuntimeError::new_err(format!("end index out of range: sb={}, eb={}, data_size={}", sb, eb, self.data.len())));
        //         }
        //         let data = self.data[sa..ea];
        //         self.data[sa..ea]
        //             .iter_mut()
        //             .zip(self.data[sb..eb].iter())
        //             .for_each(|(x, y)| x.add_assign(y, &pk.pk));
        //     }
        //     None => {
        //         self.data[sa..]
        //             .iter_mut()
        //             .zip(self.data[sb..].iter())
        //             .for_each(|(x, y)| x.add_assign(y, &pk.pk));
        //     }
        // };
        Ok(())
    }
    fn iadd_vec(
        &mut self,
        other: &FixedpointPaillierVector,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> PyResult<()> {
        match size {
            Some(s) => {
                let ea = sa + s;
                let eb = sb + s;
                if ea > self.data.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "end index out of range: sa={}, ea={}, data_size={}",
                        sa,
                        ea,
                        self.data.len()
                    )));
                }
                if eb > other.data.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "end index out of range: sb={}, eb={}, data_size={}",
                        sb,
                        eb,
                        other.data.len()
                    )));
                }
                self.data[sa..ea]
                    .iter_mut()
                    .zip(other.data[sb..eb].iter())
                    .for_each(|(x, y)| {
                        x.add_assign(y, &pk.pk)
                    });
            }
            None => {
                self.data[sa..]
                    .iter_mut()
                    .zip(other.data[sb..].iter())
                    .for_each(|(x, y)| x.add_assign(y, &pk.pk));
            }
        };
        Ok(())
    }
    fn iadd(&mut self, pk: &PK, other: &FixedpointPaillierVector) {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| x.add_assign(y, &pk.pk));
    }
    fn idouble(&mut self, pk: &PK) {
        // TODO: fix me, remove clone
        self.data
            .iter_mut()
            .for_each(|x| x.add_assign(&x.clone(), &pk.pk));
    }
    fn chunking_cumsum_with_step(&mut self, pk: &PK, chunk_sizes: Vec<usize>, step: usize) {
        let mut placeholder = CT::zero();
        let mut i = 0;
        for chunk_size in chunk_sizes {
            for j in step..chunk_size {
                placeholder = std::mem::replace(&mut self.data[i + j], placeholder);
                placeholder.add_assign(&self.data[i + j - step], &pk.pk);
                placeholder = std::mem::replace(&mut self.data[i + j], placeholder);
            }
            i += chunk_size;
        }
    }
    fn intervals_sum_with_step(
        &mut self,
        pk: &PK,
        intervals: Vec<(usize, usize)>,
        step: usize,
    ) -> FixedpointPaillierVector {
        let mut data = vec![CT::zero(); intervals.len() * step];
        for (i, (s, e)) in intervals.iter().enumerate() {
            let chunk = &mut data[i * step..(i + 1) * step];
            let sub_vec = &self.data[*s..*e];
            for (val, c) in sub_vec.iter().zip((0..step).cycle()) {
                chunk[c].add_assign(val, &pk.pk);
            }
        }
        FixedpointPaillierVector { data }
    }

    fn tolist(&self) -> Vec<PyCT> {
        self.data.iter().map(|x| PyCT { ct: x.clone() }).collect()
    }

    fn add(&self, pk: &PK, other: &FixedpointPaillierVector) -> FixedpointPaillierVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.add(y, &pk.pk))
            .collect();
        FixedpointPaillierVector { data }
    }
    fn add_scalar(&self, pk: &PK, other: &PyCT) -> FixedpointPaillierVector {
        let data = self.data.iter().map(|x| x.add(&other.ct, &pk.pk)).collect();
        FixedpointPaillierVector { data }
    }
    fn sub(&self, pk: &PK, other: &FixedpointPaillierVector) -> FixedpointPaillierVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.sub(y, &pk.pk))
            .collect();
        FixedpointPaillierVector { data }
    }
    fn sub_scalar(&self, pk: &PK, other: &PyCT) -> FixedpointPaillierVector {
        let data = self.data.iter().map(|x| x.sub(&other.ct, &pk.pk)).collect();
        FixedpointPaillierVector { data }
    }
    fn rsub(&self, pk: &PK, other: &FixedpointPaillierVector) -> FixedpointPaillierVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| y.sub(x, &pk.pk))
            .collect();
        FixedpointPaillierVector { data }
    }
    fn rsub_scalar(&self, pk: &PK, other: &PyCT) -> FixedpointPaillierVector {
        let data = self.data.iter().map(|x| other.ct.sub(x, &pk.pk)).collect();
        FixedpointPaillierVector { data }
    }
    fn mul(&self, pk: &PK, other: &FixedpointVector) -> FixedpointPaillierVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.mul(y, &pk.pk))
            .collect();
        FixedpointPaillierVector { data }
    }
    fn mul_scalar(&self, pk: &PK, other: &FixedpointEncoded) -> FixedpointPaillierVector {
        let data = self
            .data
            .iter()
            .map(|x| x.mul(&other.data, &pk.pk))
            .collect();
        FixedpointPaillierVector { data }
    }

    fn matmul(
        &self,
        pk: &PK,
        other: &FixedpointVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
    ) -> FixedpointPaillierVector {
        let mut data = vec![CT::zero(); lshape[0] * rshape[1]];
        for i in 0..lshape[0] {
            for j in 0..rshape[1] {
                for k in 0..lshape[1] {
                    data[i * rshape[1] + j].add_assign(
                        &self.data[i * lshape[1] + k].mul(&other.data[k * rshape[1] + j], &pk.pk),
                        &pk.pk,
                    );
                }
            }
        }
        FixedpointPaillierVector { data }
    }

    fn rmatmul(
        &self,
        pk: &PK,
        other: &FixedpointVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
    ) -> FixedpointPaillierVector {
        // rshape, lshape -> rshape[0] x lshape[1]
        // other, self
        // 4 x 2, 2 x 5
        // ik, kj  -> ij
        let mut data = vec![CT::zero(); lshape[1] * rshape[0]];
        for i in 0..rshape[0] {
            // 4
            for j in 0..lshape[1] {
                // 5
                for k in 0..rshape[1] {
                    // 2
                    data[i * lshape[1] + j].add_assign(
                        &self.data[k * lshape[1] + j].mul(&other.data[i * rshape[1] + k], &pk.pk),
                        &pk.pk,
                    );
                }
            }
        }
        FixedpointPaillierVector { data }
    }
}

#[pymethods]
impl FixedpointVector {
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(FixedpointVector { data: vec![] })
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.data).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.data = bincode::deserialize(&state).unwrap();
        Ok(())
    }
    fn __str__(&self) -> String {
        format!("{:?}", self.data)
    }
    fn get_stride(&mut self, index: usize, stride: usize) -> FixedpointVector {
        let start = index * stride;
        let end = start + stride;
        let data = self.data[start..end].to_vec();
        FixedpointVector { data }
    }
    fn tolist(&self) -> Vec<FixedpointEncoded> {
        self.data
            .iter()
            .map(|x| FixedpointEncoded { data: x.clone() })
            .collect()
    }
}

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FixedpointPaillierVector>()?;
    m.add_class::<FixedpointVector>()?;
    m.add_class::<PK>()?;
    m.add_class::<SK>()?;
    m.add_class::<Coders>()?;
    m.add_class::<PyCT>()?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    Ok(())
}
