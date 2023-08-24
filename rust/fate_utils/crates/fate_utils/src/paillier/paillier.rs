use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use anyhow::Error as AnyhowError;

trait ToPyErr {
    fn to_py_err(self) -> PyErr;
}

impl ToPyErr for AnyhowError {
    fn to_py_err(self) -> PyErr {
        PyRuntimeError::new_err(self.to_string())
    }
}


#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct PK(fixedpoint_paillier::PK);

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct SK(fixedpoint_paillier::SK);

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct Coder(fixedpoint_paillier::Coder);

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct Ciphertext(fixedpoint_paillier::Ciphertext);

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default, Debug)]
pub struct CiphertextVector(fixedpoint_paillier::CiphertextVector);

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default, Debug)]
pub struct PlaintextVector(fixedpoint_paillier::PlaintextVector);

#[pyclass(module = "fate_utils.paillier")]
#[derive(Default)]
pub struct Plaintext(fixedpoint_paillier::Plaintext);

#[pyclass]
pub struct Evaluator {}

#[pymethods]
impl PK {
    fn encrypt_encoded(
        &self,
        plaintext_vector: &PlaintextVector,
        obfuscate: bool,
    ) -> CiphertextVector {
        CiphertextVector(self.0.encrypt_encoded(&plaintext_vector.0, obfuscate))
    }
    fn encrypt_encoded_scalar(&self, plaintext: &Plaintext, obfuscate: bool) -> Ciphertext {
        Ciphertext(self.0.encrypt_encoded_scalar(&plaintext.0, obfuscate))
    }

    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(PK::default())
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.0).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state).unwrap();
        Ok(())
    }
}

#[pymethods]
impl SK {
    fn decrypt_to_encoded(&self, data: &CiphertextVector) -> PlaintextVector {
        PlaintextVector(self.0.decrypt_to_encoded(&data.0))
    }
    fn decrypt_to_encoded_scalar(&self, data: &Ciphertext) -> Plaintext {
        Plaintext(self.0.decrypt_to_encoded_scalar(&data.0))
    }

    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(SK::default())
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.0).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state).unwrap();
        Ok(())
    }
}

#[pymethods]
impl Coder {
    fn encode_f64(&self, data: f64) -> Plaintext {
        Plaintext(self.0.encode_f64(data))
    }
    fn decode_f64(&self, data: &Plaintext) -> f64 {
        self.0.decode_f64(&data.0)
    }
    fn encode_f32(&self, data: f32) -> Plaintext {
        Plaintext(self.0.encode_f32(data))
    }
    fn encode_i64(&self, data: i64) -> Plaintext {
        Plaintext(self.0.encode_i64(data))
    }
    fn decode_i64(&self, data: &Plaintext) -> i64 {
        self.0.decode_i64(&data.0)
    }
    fn encode_i32(&self, data: i32) -> Plaintext {
        Plaintext(self.0.encode_i32(data))
    }
    fn decode_i32(&self, data: &Plaintext) -> i32 {
        self.0.decode_i32(&data.0)
    }
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(Coder::default())
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.0).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state).unwrap();
        Ok(())
    }

    fn pack_floats(&self, float_tensor: Vec<f64>, offset_bit: usize, pack_num: usize, precision: u32) -> PlaintextVector {
        let int_scale = 2_u32.pow(precision) as f64;
        let data = float_tensor.iter().map(|x| (x * int_scale) as u64).collect::<Vec<u64>>()
            .chunks(pack_num)
            .map(|x| self.0.pack(x, offset_bit))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
    }

    fn unpack_floats(&self, packed: &PlaintextVector, offset_bit: usize, pack_num: usize, precision: u32, total_num: usize) -> Vec<f64> {
        let int_scale = 2_u32.pow(precision) as f64;
        let mut result = Vec::with_capacity(total_num);
        let mut total_num = total_num;
        for x in packed.0.data.iter() {
            let n = std::cmp::min(total_num, pack_num);
            result.extend(self.0.unpack(x, offset_bit, n).iter().map(|x| (*x as f64) / int_scale));
            total_num -= n;
        }
        result
    }
    fn encode_f64_vec(&self, data: PyReadonlyArray1<f64>) -> PlaintextVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.0.encode_f64(*x))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
    }
    fn pack_u64_vec(&self, data: Vec<u64>, shift_bit: usize, num_each_pack: usize) -> PlaintextVector {
        PlaintextVector(fixedpoint_paillier::PlaintextVector {
            data:
            data.chunks(num_each_pack).map(|x| {
                self.0.pack(x, shift_bit)
            }).collect::<Vec<_>>()
        })
    }
    fn unpack_u64_vec(&self, data: &PlaintextVector, shift_bit: usize, num_each_pack: usize, total_num: usize) -> Vec<u64> {
        let mut result = Vec::with_capacity(total_num);
        let mut total_num = total_num;
        for x in data.0.data.iter() {
            let n = std::cmp::min(total_num, num_each_pack);
            result.extend(self.0.unpack(x, shift_bit, n));
            total_num -= n;
        }
        result
    }
    fn decode_f64_vec<'py>(&self, data: &PlaintextVector, py: Python<'py>) -> &'py PyArray1<f64> {
        Array1::from(
            data.0.data
                .iter()
                .map(|x| self.0.decode_f64(x))
                .collect::<Vec<f64>>(),
        )
            .into_pyarray(py)
    }
    fn encode_f32_vec(&self, data: PyReadonlyArray1<f32>) -> PlaintextVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.0.encode_f32(*x))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
    }
    fn decode_f32(&self, data: &Plaintext) -> f32 {
        self.0.decode_f32(&data.0)
    }
    fn decode_f32_vec<'py>(&self, data: &PlaintextVector, py: Python<'py>) -> &'py PyArray1<f32> {
        Array1::from(
            data.0.data
                .iter()
                .map(|x| self.0.decode_f32(x))
                .collect::<Vec<f32>>(),
        )
            .into_pyarray(py)
    }
    fn encode_i64_vec(&self, data: PyReadonlyArray1<i64>) -> PlaintextVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.0.encode_i64(*x))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
    }
    fn decode_i64_vec(&self, data: &PlaintextVector) -> Vec<i64> {
        data.0.data.iter().map(|x| self.0.decode_i64(x)).collect()
    }
    fn encode_i32_vec(&self, data: PyReadonlyArray1<i32>) -> PlaintextVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.0.encode_i32(*x))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
    }
    fn decode_i32_vec(&self, data: &PlaintextVector) -> Vec<i32> {
        data.0.data.iter().map(|x| self.0.decode_i32(x)).collect()
    }
}

#[pyfunction]
fn keygen(bit_length: u32) -> (SK, PK, Coder) {
    let (sk, pk, coder) = fixedpoint_paillier::keygen(bit_length);
    (SK(sk), PK(pk), Coder(coder))
}

#[pymethods]
impl CiphertextVector {
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector { data: vec![] }))
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.0).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state).unwrap();
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.0.data.len()
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    #[staticmethod]
    pub fn zeros(size: usize) -> PyResult<Self> {
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector::zeros(size)))
    }

    pub fn pack_squeeze(&self, pack_num: usize, offset_bit: u32, pk: &PK) -> PyResult<CiphertextVector> {
        Ok(CiphertextVector(self.0.pack_squeeze(&pk.0, pack_num, offset_bit)))
    }

    fn slice(&mut self, start: usize, size: usize) -> CiphertextVector {
        CiphertextVector(self.0.slice(start, size))
    }

    fn slice_indexes(&mut self, indexes: Vec<usize>) -> PyResult<Self> {
        Ok(CiphertextVector(self.0.slice_indexes(indexes)))
    }
    pub fn cat(&self, others: Vec<PyRef<CiphertextVector>>) -> PyResult<Self> {
        Ok(CiphertextVector(self.0.cat(others.iter().map(|x| &x.0).collect())))
    }
    fn i_shuffle(&mut self, indexes: Vec<usize>) {
        self.0.i_shuffle(indexes);
    }
    fn intervals_slice(&mut self, intervals: Vec<(usize, usize)>) -> PyResult<Self> {
        Ok(CiphertextVector(self.0.intervals_slice(intervals).map_err(|e| e.to_py_err())?))
    }
    fn iadd_slice(&mut self, pk: &PK, position: usize, other: Vec<PyRef<Ciphertext>>) {
        self.0.iadd_slice(&pk.0, position, other.iter().map(|x| &x.0).collect());
    }
    fn iadd_vec_self(
        &mut self,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> PyResult<()> {
        self.0.iadd_vec_self(sa, sb, size, &pk.0).map_err(|e| e.to_py_err())?;
        Ok(())
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
    // }
    fn iadd_vec(
        &mut self,
        other: &CiphertextVector,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> PyResult<()> {
        self.0.iadd_vec(&other.0, sa, sb, size, &pk.0).map_err(|e| e.to_py_err())?;
        Ok(())
    }

    fn iupdate(&mut self, other: &CiphertextVector, indexes: Vec<Vec<usize>>, stride: usize, pk: &PK) -> PyResult<()> {
        self.0.iupdate(&other.0, indexes, stride, &pk.0).map_err(|e| e.to_py_err())?;
        Ok(())
    }
    fn iadd(&mut self, pk: &PK, other: &CiphertextVector) {
        self.0.iadd(&pk.0, &other.0);
    }
    fn idouble(&mut self, pk: &PK) {
        self.0.idouble(&pk.0);
    }
    fn chunking_cumsum_with_step(&mut self, pk: &PK, chunk_sizes: Vec<usize>, step: usize) {
        self.0.chunking_cumsum_with_step(&pk.0, chunk_sizes, step);
    }
    fn intervals_sum_with_step(
        &mut self,
        pk: &PK,
        intervals: Vec<(usize, usize)>,
        step: usize,
    ) -> CiphertextVector {
        CiphertextVector(self.0.intervals_sum_with_step(&pk.0, intervals, step))
    }

    fn tolist(&self) -> Vec<Ciphertext> {
        self.0.tolist().iter().map(|x| Ciphertext(x.clone())).collect()
    }

    fn add(&self, pk: &PK, other: &CiphertextVector) -> CiphertextVector {
        CiphertextVector(self.0.add(&pk.0, &other.0))
    }
    fn add_scalar(&self, pk: &PK, other: &Ciphertext) -> CiphertextVector {
        CiphertextVector(self.0.add_scalar(&pk.0, &other.0))
    }
    fn sub(&self, pk: &PK, other: &CiphertextVector) -> CiphertextVector {
        CiphertextVector(self.0.sub(&pk.0, &other.0))
    }
    fn sub_scalar(&self, pk: &PK, other: &Ciphertext) -> CiphertextVector {
        CiphertextVector(self.0.sub_scalar(&pk.0, &other.0))
    }
    fn rsub(&self, pk: &PK, other: &CiphertextVector) -> CiphertextVector {
        CiphertextVector(self.0.rsub(&pk.0, &other.0))
    }
    fn rsub_scalar(&self, pk: &PK, other: &Ciphertext) -> CiphertextVector {
        CiphertextVector(self.0.rsub_scalar(&pk.0, &other.0))
    }
    fn mul(&self, pk: &PK, other: &PlaintextVector) -> CiphertextVector {
        CiphertextVector(self.0.mul(&pk.0, &other.0))
    }
    fn mul_scalar(&self, pk: &PK, other: &Plaintext) -> CiphertextVector {
        CiphertextVector(self.0.mul_scalar(&pk.0, &other.0))
    }

    fn matmul(
        &self,
        pk: &PK,
        other: &PlaintextVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
    ) -> CiphertextVector {
        CiphertextVector(self.0.matmul(&pk.0, &other.0, lshape, rshape))
    }

    fn rmatmul(
        &self,
        pk: &PK,
        other: &PlaintextVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
    ) -> CiphertextVector {
        CiphertextVector(self.0.rmatmul(&pk.0, &other.0, lshape, rshape))
    }
}

#[pymethods]
impl PlaintextVector {
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(PlaintextVector(fixedpoint_paillier::PlaintextVector { data: vec![] }))
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.0).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state).unwrap();
        Ok(())
    }
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn get_stride(&mut self, index: usize, stride: usize) -> PlaintextVector {
        PlaintextVector(self.0.get_stride(index, stride))
    }
    fn tolist(&self) -> Vec<Plaintext> {
        self.0.tolist().iter().map(|x| Plaintext(x.clone())).collect()
    }
}

#[pymethods]
impl Evaluator {
    #[staticmethod]
    fn cat(vec_list: Vec<PyRef<CiphertextVector>>) -> PyResult<CiphertextVector> {
        let mut data = vec![fixedpoint_paillier::Ciphertext::zero(); 0];
        for vec in vec_list {
            data.extend(vec.0.data.clone());
        }
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector { data }))
    }
    #[staticmethod]
    fn slice_indexes(a: &CiphertextVector, indexes: Vec<usize>) -> PyResult<CiphertextVector> {
        let data = indexes
            .iter()
            .map(|i| a.0.data[*i].clone())
            .collect::<Vec<_>>();
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector { data }))
    }
}

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CiphertextVector>()?;
    m.add_class::<PlaintextVector>()?;
    m.add_class::<PK>()?;
    m.add_class::<SK>()?;
    m.add_class::<Coder>()?;
    m.add_class::<Ciphertext>()?;
    m.add_class::<Evaluator>()?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    Ok(())
}
