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
pub struct Packer(fixedpoint_paillier::Coder);

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
pub struct HistEvaluator {}

#[pymethods]
impl PK {
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
impl Packer {
    #[staticmethod]
    fn pack_floats(float_tensor: Vec<f64>, offset_bit: usize, pack_num: usize, precision: u32) -> PlaintextVector {
        let int_scale = 2_u32.pow(precision) as f64;
        let data = float_tensor.iter().map(|x| (x * int_scale) as u64).collect::<Vec<u64>>()
            .chunks(pack_num)
            .map(|x| fixedpoint_paillier::Packer::pack(x, offset_bit))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
    }
    #[staticmethod]
    fn unpack_floats(packed: &PlaintextVector, offset_bit: usize, pack_num: usize, precision: u32, total_num: usize) -> Vec<f64> {
        let int_scale = 2_u32.pow(precision) as f64;
        let mut result = Vec::with_capacity(total_num);
        let mut total_num = total_num;
        for x in packed.0.data.iter() {
            let n = std::cmp::min(total_num, pack_num);
            result.extend(fixedpoint_paillier::Packer::unpack(x, offset_bit, n).iter().map(|x| (*x as f64) / int_scale));
            total_num -= n;
        }
        result
    }
    #[staticmethod]
    fn pack_u64_vec(data: Vec<u64>, shift_bit: usize, num_each_pack: usize) -> PlaintextVector {
        PlaintextVector(fixedpoint_paillier::PlaintextVector {
            data:
            data.chunks(num_each_pack).map(|x| {
                fixedpoint_paillier::Packer::pack(x, shift_bit)
            }).collect::<Vec<_>>()
        })
    }
    #[staticmethod]
    fn unpack_u64_vec(data: &PlaintextVector, shift_bit: usize, num_each_pack: usize, total_num: usize) -> Vec<u64> {
        let mut result = Vec::with_capacity(total_num);
        let mut total_num = total_num;
        for x in data.0.data.iter() {
            let n = std::cmp::min(total_num, num_each_pack);
            result.extend(fixedpoint_paillier::Packer::unpack(x, shift_bit, n));
            total_num -= n;
        }
        result
    }
}

#[pymethods]
impl Coder {
    #[staticmethod]
    fn from_pk(pk: &PK) -> Self {
        Coder(fixedpoint_paillier::Coder::from_pk(&pk.0))
    }
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

    fn encode_f64_vec(&self, data: PyReadonlyArray1<f64>) -> PlaintextVector {
        let data = data
            .as_array()
            .iter()
            .map(|x| self.0.encode_f64(*x))
            .collect();
        PlaintextVector(fixedpoint_paillier::PlaintextVector { data })
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
fn keygen(bit_length: u32) -> (SK, PK) {
    let (sk, pk, coder) = fixedpoint_paillier::keygen(bit_length);
    (SK(sk), PK(pk))
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

    fn slice_indexes(&self, indexes: Vec<usize>) -> PyResult<CiphertextVector> {
        let data = indexes
            .iter()
            .map(|i| self.0.data[*i].clone())
            .collect::<Vec<_>>();
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector { data }))
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


#[pyclass]
pub struct BaseEvaluator {}

#[pymethods]
impl BaseEvaluator {
    #[staticmethod]
    fn encrypt(
        plaintext_vector: &PlaintextVector,
        obfuscate: bool,
        pk: &PK,
    ) -> CiphertextVector {
        CiphertextVector(pk.0.encrypt_encoded(&plaintext_vector.0, obfuscate))
    }
    #[staticmethod]
    fn encrypt_scalar(plaintext: &Plaintext, obfuscate: bool, pk: &PK) -> Ciphertext {
        Ciphertext(pk.0.encrypt_encoded_scalar(&plaintext.0, obfuscate))
    }

    #[staticmethod]
    fn decrypt(data: &CiphertextVector, sk: &SK) -> PlaintextVector {
        PlaintextVector(sk.0.decrypt_to_encoded(&data.0))
    }
    #[staticmethod]
    fn decrypt_scalar(data: &Ciphertext, sk: &SK) -> Plaintext {
        Plaintext(sk.0.decrypt_to_encoded_scalar(&data.0))
    }
    #[staticmethod]
    fn cat(vec_list: Vec<PyRef<CiphertextVector>>) -> PyResult<CiphertextVector> {
        let mut data = vec![fixedpoint_paillier::Ciphertext::zero(); 0];
        for vec in vec_list {
            data.extend(vec.0.data.clone());
        }
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector { data }))
    }
}

#[pyclass]
pub struct HistogramEvaluator {}

#[pymethods]
impl HistogramEvaluator {
    #[staticmethod]
    fn slice_indexes(a: &CiphertextVector, indexes: Vec<usize>) -> PyResult<CiphertextVector> {
        let data = indexes
            .iter()
            .map(|i| a.0.data[*i].clone())
            .collect::<Vec<_>>();
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector { data }))
    }
    #[staticmethod]
    fn intervals_sum_with_step(
        a: &mut CiphertextVector,
        intervals: Vec<(usize, usize)>,
        step: usize,
        pk: &PK,
    ) -> CiphertextVector {
        CiphertextVector(a.0.intervals_sum_with_step(&pk.0, intervals, step))
    }

    #[staticmethod]
    fn iupdate(a: &mut CiphertextVector, b: &CiphertextVector, indexes: Vec<Vec<usize>>, stride: usize, pk: &PK) -> PyResult<()> {
        a.0.iupdate(&b.0, indexes, stride, &pk.0).map_err(|e| e.to_py_err())?;
        Ok(())
    }
    #[staticmethod]
    fn iadd_vec(
        a: &mut CiphertextVector,
        b: &CiphertextVector,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> PyResult<()> {
        a.0.iadd_vec(&b.0, sa, sb, size, &pk.0).map_err(|e| e.to_py_err())?;
        Ok(())
    }
    #[staticmethod]
    fn chunking_cumsum_with_step(a: &mut CiphertextVector, chunk_sizes: Vec<usize>, step: usize, pk: &PK) {
        a.0.chunking_cumsum_with_step(&pk.0, chunk_sizes, step);
    }
    #[staticmethod]
    fn tolist(a: &CiphertextVector) -> Vec<Ciphertext> {
        a.0.tolist().iter().map(|x| Ciphertext(x.clone())).collect()
    }
    #[staticmethod]
    pub fn pack_squeeze(a: &CiphertextVector, pack_num: usize, offset_bit: u32, pk: &PK) -> PyResult<CiphertextVector> {
        Ok(CiphertextVector(a.0.pack_squeeze(&pk.0, pack_num, offset_bit)))
    }
    #[staticmethod]
    fn slice(a: &mut CiphertextVector, start: usize, size: usize) -> CiphertextVector {
        CiphertextVector(a.0.slice(start, size))
    }

    #[staticmethod]
    fn i_shuffle(a: &mut CiphertextVector, indexes: Vec<usize>) {
        a.0.i_shuffle(indexes);
    }
    #[staticmethod]
    fn intervals_slice(a: &mut CiphertextVector, intervals: Vec<(usize, usize)>) -> PyResult<CiphertextVector> {
        Ok(CiphertextVector(a.0.intervals_slice(intervals).map_err(|e| e.to_py_err())?))
    }
    #[staticmethod]
    fn iadd_slice(a: &mut CiphertextVector, b: Vec<PyRef<Ciphertext>>, position: usize, pk: &PK) {
        a.0.iadd_slice(&pk.0, position, b.iter().map(|x| &x.0).collect());
    }
    #[staticmethod]
    fn iadd_vec_self(
        a: &mut CiphertextVector,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> PyResult<()> {
        a.0.iadd_vec_self(sa, sb, size, &pk.0).map_err(|e| e.to_py_err())?;
        Ok(())
    }
}

#[pyclass]
pub struct ArithmeticEvaluator {}

#[pymethods]
impl ArithmeticEvaluator {
    #[staticmethod]
    pub fn zeros(size: usize) -> PyResult<CiphertextVector> {
        Ok(CiphertextVector(fixedpoint_paillier::CiphertextVector::zeros(size)))
    }
    #[staticmethod]
    fn iadd(a: &mut CiphertextVector, pk: &PK, other: &CiphertextVector) {
        a.0.iadd(&pk.0, &other.0);
    }
    #[staticmethod]
    fn idouble(a: &mut CiphertextVector, pk: &PK) {
        a.0.idouble(&pk.0);
    }
    #[staticmethod]
    fn add(a: &CiphertextVector, other: &CiphertextVector, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.add(&pk.0, &other.0))
    }
    #[staticmethod]
    fn add_scalar(a: &CiphertextVector, other: &Ciphertext, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.add_scalar(&pk.0, &other.0))
    }
    #[staticmethod]
    fn sub(a: &CiphertextVector, other: &CiphertextVector, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.sub(&pk.0, &other.0))
    }
    #[staticmethod]
    fn sub_scalar(a: &CiphertextVector, other: &Ciphertext, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.sub_scalar(&pk.0, &other.0))
    }
    #[staticmethod]
    fn rsub(a: &CiphertextVector, other: &CiphertextVector, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.rsub(&pk.0, &other.0))
    }
    #[staticmethod]
    fn rsub_scalar(a: &CiphertextVector, other: &Ciphertext, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.rsub_scalar(&pk.0, &other.0))
    }
    #[staticmethod]
    fn mul(a: &CiphertextVector, other: &PlaintextVector, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.mul(&pk.0, &other.0))
    }
    #[staticmethod]
    fn mul_scalar(a: &CiphertextVector, other: &Plaintext, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.mul_scalar(&pk.0, &other.0))
    }

    #[staticmethod]
    fn sum(a: &CiphertextVector, shape: Vec<usize>, dim: Option<usize>, pk: &PK) -> CiphertextVector {
        CiphertextVector(a.0.sum(shape, dim, &pk.0))
    }
    #[staticmethod]
    fn matmul(
        a: &CiphertextVector,
        other: &PlaintextVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
        pk: &PK,
    ) -> CiphertextVector {
        CiphertextVector(a.0.matmul(&pk.0, &other.0, lshape, rshape))
    }
    #[staticmethod]
    fn rmatmul(
        a: &CiphertextVector,
        other: &PlaintextVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
        pk: &PK,
    ) -> CiphertextVector {
        CiphertextVector(a.0.rmatmul(&pk.0, &other.0, lshape, rshape))
    }
}

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CiphertextVector>()?;
    m.add_class::<PlaintextVector>()?;
    m.add_class::<PK>()?;
    m.add_class::<SK>()?;
    m.add_class::<Coder>()?;
    m.add_class::<Packer>()?;
    m.add_class::<Ciphertext>()?;
    m.add_class::<Plaintext>()?;
    m.add_class::<BaseEvaluator>()?;
    m.add_class::<ArithmeticEvaluator>()?;
    m.add_class::<HistogramEvaluator>()?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    Ok(())
}
