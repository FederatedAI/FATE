use math::BInt;
use paillier;
use anyhow::Result;
use anyhow::anyhow;
use std::ops::{AddAssign, BitAnd, Mul, ShlAssign, SubAssign};
use rug::{self, Integer, ops::Pow, Float, Rational};
use serde::{Deserialize, Serialize};

mod frexp;

use frexp::Frexp;

const BASE: u32 = 16;
const MAX_INT_FRACTION: u8 = 2;
const FLOAT_MANTISSA_BITS: u32 = 53;
const LOG2_BASE: u32 = 4;

#[derive(Default, Serialize, Deserialize)]
pub struct PK {
    pub pk: paillier::PK,
    pub max_int: BInt,
}

impl PK {
    #[inline]
    pub fn encrypt(&self, plaintext: &Plaintext, obfuscate: bool) -> Ciphertext {
        let exp = plaintext.exp;
        let encode = self.pk.encrypt(&plaintext.significant, obfuscate);
        Ciphertext {
            significant_encryped: encode,
            exp,
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct SK {
    pub sk: paillier::SK,
}

impl SK {
    #[inline]
    pub fn decrypt(&self, ciphertext: &Ciphertext) -> Plaintext {
        let exp = ciphertext.exp;
        Plaintext {
            significant: self.sk.decrypt(&ciphertext.significant_encryped),
            exp,
        }
    }
}


/// fixedpoint encoder
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Coder {
    pub n: BInt,
    pub max_int: BInt,
}

impl Coder {
    pub fn new(n: &BInt) -> Self {
        Coder {
            n: n.clone(),
            max_int: n / MAX_INT_FRACTION,
        }
    }

    pub fn encode_i64(&self, plaintext: i64) -> Plaintext {
        let significant = paillier::PT(if plaintext < 0 {
            BInt::from(&self.n + plaintext)
        } else {
            BInt::from(plaintext)
        });
        Plaintext {
            significant,
            exp: 0,
        }
    }
    pub fn pack_floats(&self, floats: &Vec<f64>, offset_bit: usize, pack_num: usize, precision: u32) -> Vec<Plaintext> {
        let int_scale = Integer::from(2).pow(precision);
        floats.chunks(pack_num).map(|data| {
            let significant = data.iter().fold(Integer::default(), |mut x, v| {
                x.shl_assign(offset_bit);
                x.add_assign(Float::with_val(64, v).mul(&int_scale).round().to_integer().unwrap());
                x
            });
            Plaintext {
                significant: paillier::PT(BInt(significant)),
                exp: 0,
            }
        })
            .collect()
    }
    pub fn unpack_floats(&self, encoded: &[Plaintext], offset_bit: usize, pack_num: usize, precision: u32, expect_total_num: usize) -> Vec<f64> {
        let int_scale = Integer::from(2).pow(precision);
        let mut mask = Integer::from(1);
        mask <<= offset_bit;
        mask.sub_assign(1);
        let mut result = Vec::with_capacity(expect_total_num);
        let mut total_num = expect_total_num;
        for x in encoded {
            let n = std::cmp::min(total_num, pack_num);
            let mut significant = x.significant.0.0.clone();
            let mut temp = Vec::with_capacity(n);
            for _ in 0..n {
                let value = Rational::from(((&significant).bitand(&mask), &int_scale)).to_f64();
                temp.push(value);
                significant >>= offset_bit;
            }
            temp.reverse();
            result.extend(temp);
            total_num -= n;
        }
        #[cfg(debug_assertions)]
        assert_eq!(result.len(), expect_total_num);

        result
    }
    pub fn encode_i32(&self, plaintext: i32) -> Plaintext {
        let significant = paillier::PT(if plaintext < 0 {
            BInt::from(&self.n + plaintext)
        } else {
            BInt::from(plaintext)
        });
        Plaintext {
            significant,
            exp: 0,
        }
    }
    pub fn decode_i64(&self, encoded: &Plaintext) -> i64 {
        let significant = encoded.significant.0.clone();
        let mantissa = if significant > self.n {
            panic!("Attempted to decode corrupted number")
        } else if significant <= self.max_int {
            significant
        } else if significant >= BInt::from(&self.n - &self.max_int) {
            significant - &self.n
        } else {
            panic!("Overflow detected in decrypted number")
        };
        (mantissa << (LOG2_BASE as i32 * encoded.exp)).to_i128() as i64
    }
    pub fn decode_i32(&self, encoded: &Plaintext) -> i32 {
        // Todo: could be improved
        self.decode_f64(encoded) as i32
    }

    pub fn encode_f64(&self, plaintext: f64) -> Plaintext {
        let bin_flt_exponent = plaintext.frexp().1;
        let bin_lsb_exponent = bin_flt_exponent - (FLOAT_MANTISSA_BITS as i32);
        let exp = (bin_lsb_exponent as f64 / LOG2_BASE as f64).floor() as i32;
        let significant = BInt(
            (plaintext * rug::Float::with_val(FLOAT_MANTISSA_BITS, BASE).pow(-exp))
                .round()
                .to_integer()
                .unwrap(),
        );
        if significant.abs_ref() > self.max_int {
            panic!(
                "Integer needs to be within +/- {} but got {}",
                self.max_int.0, &significant.0
            )
        }
        Plaintext {
            significant: paillier::PT(significant),
            exp,
        }
    }
    pub fn decode_f64(&self, encoded: &Plaintext) -> f64 {
        let significant = encoded.significant.0.clone();
        let mantissa = if significant > self.n {
            panic!("Attempted to decode corrupted number")
        } else if significant <= self.max_int {
            significant
        } else if significant >= BInt::from(&self.n - &self.max_int) {
            significant - &self.n
        } else {
            format!("Overflow detected in decrypted number: {:?}", significant);
            panic!("Overflow detected in decrypted number")
        };
        if encoded.exp >= 0 {
            (mantissa << (LOG2_BASE as i32 * encoded.exp)).to_f64()
        } else {
            (mantissa * rug::Float::with_val(FLOAT_MANTISSA_BITS, BASE).pow(encoded.exp)).to_f64()
        }
    }
    pub fn encode_f32(&self, plaintext: f32) -> Plaintext {
        self.encode_f64(plaintext as f64)
    }
    pub fn decode_f32(&self, encoded: &Plaintext) -> f32 {
        self.decode_f64(encoded) as f32
    }
}

pub trait CouldCode {
    fn encode(&self, coder: &Coder) -> Plaintext;
    fn decode(plaintext: &Plaintext, coder: &Coder) -> Self;
}

impl CouldCode for f64 {
    fn encode(&self, coder: &Coder) -> Plaintext {
        coder.encode_f64(*self)
    }
    fn decode(plaintext: &Plaintext, coder: &Coder) -> Self {
        coder.decode_f64(plaintext)
    }
}

impl CouldCode for i64 {
    fn encode(&self, coder: &Coder) -> Plaintext {
        coder.encode_i64(*self)
    }
    fn decode(plaintext: &Plaintext, coder: &Coder) -> Self {
        coder.decode_i64(plaintext)
    }
}

impl CouldCode for i32 {
    fn encode(&self, coder: &Coder) -> Plaintext {
        coder.encode_i32(*self)
    }
    fn decode(plaintext: &Plaintext, coder: &Coder) -> Self {
        coder.decode_i32(plaintext)
    }
}

impl CouldCode for f32 {
    fn encode(&self, coder: &Coder) -> Plaintext {
        coder.encode_f32(*self)
    }
    fn decode(plaintext: &Plaintext, coder: &Coder) -> Self {
        coder.decode_f32(plaintext)
    }
}


#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Ciphertext {
    pub significant_encryped: paillier::CT,
    pub exp: i32,
}

impl Ciphertext {
    pub fn zero() -> Ciphertext {
        Ciphertext {
            significant_encryped: paillier::CT::zero(),
            exp: 0,
        }
    }
    fn decrese_exp_to(&self, exp: i32, pk: &paillier::PK) -> Ciphertext {
        assert!(exp < self.exp);
        let factor = BInt::from(BASE).pow((self.exp - exp) as u32);
        let significant_encryped = self.significant_encryped.mul_pt(&paillier::PT(factor), pk);
        Ciphertext {
            significant_encryped,
            exp,
        }
    }
    pub fn neg(&self, pk: &PK) -> Ciphertext {
        Ciphertext {
            significant_encryped: paillier::CT(self.significant_encryped.0.invert_ref(&pk.pk.ns)),
            exp: self.exp,
        }
    }
    pub fn add_pt(&self, b: &Plaintext, pk: &PK) -> Ciphertext {
        let b = pk.encrypt(b, false);
        self.add(&b, pk)
    }
    pub fn sub_pt(&self, b: &Plaintext, pk: &PK) -> Ciphertext {
        let b = pk.encrypt(b, false);
        self.sub(&b, pk)
    }
    /*
    other - self
    */
    pub fn rsub_pt(&self, b: &Plaintext, pk: &PK) -> Ciphertext {
        let b = pk.encrypt(b, false);
        b.sub(self, pk)
    }
    pub fn sub(&self, b: &Ciphertext, pk: &PK) -> Ciphertext {
        self.add(&b.neg(pk), pk)
    }
    pub fn rsub(&self, b: &Ciphertext, pk: &PK) -> Ciphertext {
        self.neg(pk).add(&b, pk)
    }
    pub fn add_assign(&mut self, b: &Ciphertext, pk: &PK) {
        // FIXME
        *self = self.add(b, pk);
    }
    pub fn sub_assign(&mut self, b: &Ciphertext, pk: &PK) {
        // FIXME
        *self = self.sub(b, pk);
    }
    pub fn i_double(&mut self, pk: &PK) {
        self.significant_encryped.0 = self
            .significant_encryped
            .0
            .pow_mod_ref(&BInt::from(2), &pk.pk.ns);
    }

    pub fn add(&self, b: &Ciphertext, pk: &PK) -> Ciphertext {
        let a = self;
        if a.significant_encryped.0.0 == 1 {
            return b.clone();
        }
        if b.significant_encryped.0.0 == 1 {
            return a.clone();
        }
        if a.exp > b.exp {
            let a = &a.decrese_exp_to(b.exp, &pk.pk);
            Ciphertext {
                significant_encryped: a
                    .significant_encryped
                    .add_ct(&b.significant_encryped, &pk.pk),
                exp: b.exp,
            }
        } else if a.exp < b.exp {
            let b = &b.decrese_exp_to(a.exp, &pk.pk);
            Ciphertext {
                significant_encryped: a
                    .significant_encryped
                    .add_ct(&b.significant_encryped, &pk.pk),
                exp: a.exp,
            }
        } else {
            Ciphertext {
                significant_encryped: a
                    .significant_encryped
                    .add_ct(&b.significant_encryped, &pk.pk),
                exp: a.exp,
            }
        }
    }
    pub fn mul(&self, b: &Plaintext, pk: &PK) -> Ciphertext {
        let inside = if &pk.pk.n - &pk.max_int <= b.significant.0 {
            // large plaintext
            let neg_c = self.significant_encryped.0.invert_ref(&pk.pk.ns);
            let neg_scalar = &pk.pk.n - &b.significant.0;
            neg_c.pow_mod_ref(&neg_scalar, &pk.pk.ns)
        } else if b.significant.0 <= pk.max_int {
            (&self.significant_encryped.0).pow_mod_ref(&b.significant.0, &pk.pk.ns)
        } else {
            panic!("invalid plaintext: {:?}", b)
        };
        Ciphertext {
            significant_encryped: paillier::CT(inside),
            exp: self.exp + b.exp,
        }
    }
}


#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct CiphertextVector {
    pub data: Vec<Ciphertext>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Plaintext {
    pub significant: paillier::PT,
    pub exp: i32,
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct PlaintextVector {
    pub data: Vec<Plaintext>,
}

impl PK {
    pub fn encrypt_encoded(
        &self,
        plaintext: &PlaintextVector,
        obfuscate: bool,
    ) -> CiphertextVector {
        let data = plaintext
            .data
            .iter()
            .map(|x| Ciphertext { significant_encryped: self.pk.encrypt(&x.significant, obfuscate), exp: x.exp })
            .collect();
        CiphertextVector { data }
    }
    pub fn encrypt_encoded_scalar(&self, plaintext: &Plaintext, obfuscate: bool) -> Ciphertext {
        Ciphertext {
            significant_encryped: self.pk.encrypt(&plaintext.significant, obfuscate),
            exp: plaintext.exp,
        }
    }
}


impl SK {
    pub fn decrypt_to_encoded(&self, data: &CiphertextVector) -> PlaintextVector {
        let data = data.data.iter().map(|x| Plaintext {
            significant:
            self.sk.decrypt(&x.significant_encryped),
            exp: x.exp,
        }).collect();
        PlaintextVector { data }
    }
    pub fn decrypt_to_encoded_scalar(&self, data: &Ciphertext) -> Plaintext {
        Plaintext {
            significant: self.sk.decrypt(&data.significant_encryped),
            exp: data.exp,
        }
    }
}

pub fn keygen(bit_length: u32) -> (SK, PK, Coder) {
    let (sk, pk) = paillier::keygen(bit_length);
    let coder = Coder::new(&pk.n);
    let max_int = &pk.n / MAX_INT_FRACTION;
    (SK { sk }, PK { pk: pk, max_int: max_int }, coder)
}

impl CiphertextVector {
    #[inline]
    fn iadd_i_j(&mut self, pk: &PK, i: usize, j: usize, size: usize) {
        let mut placeholder = Ciphertext::default();
        for k in 0..size {
            placeholder = std::mem::replace(&mut self.data[i + k], placeholder);
            placeholder.add_assign(&self.data[j + k], &pk);
            placeholder = std::mem::replace(&mut self.data[i + k], placeholder);
        }
    }
    #[inline]
    fn isub_i_j(&mut self, pk: &PK, i: usize, j: usize, size: usize) {
        let mut placeholder = Ciphertext::default();
        for k in 0..size {
            placeholder = std::mem::replace(&mut self.data[i + k], placeholder);
            placeholder.sub_assign(&self.data[j + k], &pk);
            placeholder = std::mem::replace(&mut self.data[i + k], placeholder);
        }
    }
    pub fn zeros(size: usize) -> Self {
        let data = vec![Ciphertext::zero(); size];
        CiphertextVector { data }
    }

    pub fn pack_squeeze(&self, pk: &PK, pack_num: usize, shift_bit: u32) -> CiphertextVector {
        let base = BInt::from(2).pow(shift_bit);
        let data = self.data.chunks(pack_num).map(|x| {
            let mut result = x[0].significant_encryped.0.clone();
            for y in &x[1..] {
                result.pow_mod_mut(&base, &pk.pk.ns);
                result = result.mul(&y.significant_encryped.0) % &pk.pk.ns;
            }
            Ciphertext { significant_encryped: paillier::CT(result), exp: 0 }
        }).collect();
        CiphertextVector { data }
    }

    pub fn slice(&mut self, start: usize, size: usize) -> CiphertextVector {
        let data = self.data[start..start + size].to_vec();
        CiphertextVector { data }
    }

    pub fn slice_indexes(&mut self, indexes: Vec<usize>) -> Self {
        let data = indexes
            .iter()
            .map(|i| self.data[*i].clone())
            .collect::<Vec<_>>();
        CiphertextVector { data }
    }

    pub fn cat(&self, others: Vec<&CiphertextVector>) -> Self {
        let mut data = self.data.clone();
        for other in others {
            data.extend(other.data.clone());
        }
        CiphertextVector { data }
    }

    pub fn i_shuffle(&mut self, indexes: Vec<usize>) {
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

    pub fn shuffle(&self, indexes: Vec<usize>) -> Self {
        let data = self.data.clone();
        let mut result = CiphertextVector { data };
        result.i_shuffle(indexes);
        result
    }

    pub fn intervals_slice(&mut self, intervals: Vec<(usize, usize)>) -> Result<Self> {
        let mut data = vec![];
        for (start, end) in intervals {
            if end > self.data.len() {
                return Err(anyhow!(
                    "end index out of range: start={}, end={}, data_size={}",
                    start,
                    end,
                    self.data.len()
                ));
            }
            data.extend_from_slice(&self.data[start..end]);
        }
        Ok(CiphertextVector { data })
    }

    pub fn iadd_slice(&mut self, pk: &PK, position: usize, other: Vec<&Ciphertext>) {
        for (i, x) in other.iter().enumerate() {
            self.data[position + i] = self.data[position + i].add(&x, &pk);
        }
    }

    pub fn iadd_vec_self(
        &mut self,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> Result<()> {
        if sa == sb {
            if let Some(s) = size {
                if sa + s > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sa={}, s={}, data_size={}",
                        sa,
                        s,
                        self.data.len()
                    ));
                }
                self.data[sa..sa + s]
                    .iter_mut()
                    .for_each(|x| x.i_double(&pk));
            } else {
                self.data[sa..].iter_mut().for_each(|x| x.i_double(&pk));
            }
        } else if sa < sb {
            // it's safe to update from left to right
            if let Some(s) = size {
                if sb + s > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sb={}, s={}, data_size={}",
                        sb,
                        s,
                        self.data.len()
                    ));
                }
                self.iadd_i_j(&pk, sb, sa, s);
            } else {
                self.iadd_i_j(&pk, sb, sa, self.data.len() - sb);
            }
        } else {
            // it's safe to update from right to left
            if let Some(s) = size {
                if sa + s > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sa={}, s={}, data_size={}",
                        sa,
                        s,
                        self.data.len()
                    ));
                }
                self.iadd_i_j(&pk, sa, sb, s);
            } else {
                self.iadd_i_j(&pk, sa, sb, self.data.len() - sa);
            }
        }
        Ok(())
    }
    pub fn isub_vec_self(
        &mut self,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> Result<()> {
        if sa == sb {
            if let Some(s) = size {
                if sa + s > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sa={}, s={}, data_size={}",
                        sa,
                        s,
                        self.data.len()
                    ));
                }
                self.data[sa..sa + s]
                    .iter_mut()
                    .for_each(|x| *x = Ciphertext::zero());
            } else {
                self.data[sa..].iter_mut().for_each(|x| *x = Ciphertext::zero());
            }
        } else if sa < sb {
            // it's safe to update from left to right
            if let Some(s) = size {
                if sb + s > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sb={}, s={}, data_size={}",
                        sb,
                        s,
                        self.data.len()
                    ));
                }
                self.isub_i_j(&pk, sb, sa, s);
            } else {
                self.isub_i_j(&pk, sb, sa, self.data.len() - sb);
            }
        } else {
            // it's safe to update from right to left
            if let Some(s) = size {
                if sa + s > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sa={}, s={}, data_size={}",
                        sa,
                        s,
                        self.data.len()
                    ));
                }
                self.isub_i_j(&pk, sa, sb, s);
            } else {
                self.isub_i_j(&pk, sa, sb, self.data.len() - sa);
            }
        }
        Ok(())
    }

    pub fn iadd_vec(
        &mut self,
        other: &CiphertextVector,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> Result<()> {
        match size {
            Some(s) => {
                let ea = sa + s;
                let eb = sb + s;
                if ea > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sa={}, ea={}, data_size={}",
                        sa,
                        ea,
                        self.data.len()
                    ));
                }
                if eb > other.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sb={}, eb={}, data_size={}",
                        sb,
                        eb,
                        other.data.len()
                    ));
                }
                self.data[sa..ea]
                    .iter_mut()
                    .zip(other.data[sb..eb].iter())
                    .for_each(|(x, y)| {
                        x.add_assign(y, &pk)
                    });
            }
            None => {
                self.data[sa..]
                    .iter_mut()
                    .zip(other.data[sb..].iter())
                    .for_each(|(x, y)| x.add_assign(y, &pk));
            }
        };
        Ok(())
    }

    pub fn isub_vec(
        &mut self,
        other: &CiphertextVector,
        sa: usize,
        sb: usize,
        size: Option<usize>,
        pk: &PK,
    ) -> Result<()> {
        match size {
            Some(s) => {
                let ea = sa + s;
                let eb = sb + s;
                if ea > self.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sa={}, ea={}, data_size={}",
                        sa,
                        ea,
                        self.data.len()
                    ));
                }
                if eb > other.data.len() {
                    return Err(anyhow!(
                        "end index out of range: sb={}, eb={}, data_size={}",
                        sb,
                        eb,
                        other.data.len()
                    ));
                }
                self.data[sa..ea]
                    .iter_mut()
                    .zip(other.data[sb..eb].iter())
                    .for_each(|(x, y)| {
                        x.sub_assign(y, &pk)
                    });
            }
            None => {
                self.data[sa..]
                    .iter_mut()
                    .zip(other.data[sb..].iter())
                    .for_each(|(x, y)| x.sub_assign(y, &pk));
            }
        };
        Ok(())
    }

    pub fn iupdate(&mut self, other: &CiphertextVector, indexes: Vec<Vec<usize>>, stride: usize, pk: &PK) -> Result<()> {
        for (i, x) in indexes.iter().enumerate() {
            let sb = i * stride;
            for pos in x.iter() {
                let sa = pos * stride;
                for i in 0..stride {
                    self.data[sa + i].add_assign(&other.data[sb + i], &pk);
                }
            }
        }
        Ok(())
    }
    pub fn iupdate_with_masks(&mut self, other: &CiphertextVector, indexes: Vec<Vec<usize>>, masks: Vec<bool>, stride: usize, pk: &PK) -> Result<()> {
        for (value_pos, x) in masks.iter().enumerate().filter(|(_, &mask)| mask).map(|(i, _)| i).zip(indexes.iter()) {
            let sb = value_pos * stride;
            for pos in x.iter() {
                let sa = pos * stride;
                for i in 0..stride {
                    self.data[sa + i].add_assign(&other.data[sb + i], &pk);
                }
            }
        }
        Ok(())
    }

    pub fn iadd(&mut self, pk: &PK, other: &CiphertextVector) {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| x.add_assign(y, &pk));
    }

    pub fn idouble(&mut self, pk: &PK) {
        // TODO: fix me, remove clone
        self.data
            .iter_mut()
            .for_each(|x| x.add_assign(&x.clone(), &pk));
    }

    pub fn chunking_cumsum_with_step(&mut self, pk: &PK, chunk_sizes: Vec<usize>, step: usize) {
        let mut placeholder = Ciphertext::zero();
        let mut i = 0;
        for chunk_size in chunk_sizes {
            for j in step..chunk_size {
                placeholder = std::mem::replace(&mut self.data[i + j], placeholder);
                placeholder.add_assign(&self.data[i + j - step], &pk);
                placeholder = std::mem::replace(&mut self.data[i + j], placeholder);
            }
            i += chunk_size;
        }
    }

    pub fn intervals_sum_with_step(
        &mut self,
        pk: &PK,
        intervals: Vec<(usize, usize)>,
        step: usize,
    ) -> CiphertextVector {
        let mut data = vec![Ciphertext::zero(); intervals.len() * step];
        for (i, (s, e)) in intervals.iter().enumerate() {
            let chunk = &mut data[i * step..(i + 1) * step];
            let sub_vec = &self.data[*s..*e];
            for (val, c) in sub_vec.iter().zip((0..step).cycle()) {
                chunk[c].add_assign(val, &pk);
            }
        }
        CiphertextVector { data }
    }

    pub fn tolist(&self) -> Vec<CiphertextVector> {
        self.data.iter().map(|x| CiphertextVector { data: vec![x.clone()] }).collect()
    }

    pub fn add(&self, pk: &PK, other: &CiphertextVector) -> CiphertextVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.add(y, &pk))
            .collect();
        CiphertextVector { data }
    }

    pub fn add_scalar(&self, pk: &PK, other: &Ciphertext) -> CiphertextVector {
        let data = self.data.iter().map(|x| x.add(&other, &pk)).collect();
        CiphertextVector { data }
    }

    pub fn sub(&self, pk: &PK, other: &CiphertextVector) -> CiphertextVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.sub(y, &pk))
            .collect();
        CiphertextVector { data }
    }

    pub fn sub_scalar(&self, pk: &PK, other: &Ciphertext) -> CiphertextVector {
        let data = self.data.iter().map(|x| x.sub(&other, &pk)).collect();
        CiphertextVector { data }
    }

    pub fn rsub(&self, pk: &PK, other: &CiphertextVector) -> CiphertextVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| y.sub(x, &pk))
            .collect();
        CiphertextVector { data }
    }

    pub fn rsub_scalar(&self, pk: &PK, other: &Ciphertext) -> CiphertextVector {
        let data = self.data.iter().map(|x| other.sub(x, &pk)).collect();
        CiphertextVector { data }
    }

    pub fn mul(&self, pk: &PK, other: &PlaintextVector) -> CiphertextVector {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.mul(y, &pk))
            .collect();
        CiphertextVector { data }
    }

    pub fn mul_scalar(&self, pk: &PK, other: &Plaintext) -> CiphertextVector {
        let data = self
            .data
            .iter()
            .map(|x| x.mul(&other, &pk))
            .collect();
        CiphertextVector { data }
    }

    pub fn matmul(
        &self,
        pk: &PK,
        other: &PlaintextVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
    ) -> CiphertextVector {
        let mut data = vec![Ciphertext::zero(); lshape[0] * rshape[1]];
        for i in 0..lshape[0] {
            for j in 0..rshape[1] {
                for k in 0..lshape[1] {
                    data[i * rshape[1] + j].add_assign(
                        &self.data[i * lshape[1] + k].mul(&other.data[k * rshape[1] + j], &pk),
                        &pk,
                    );
                }
            }
        }
        CiphertextVector { data }
    }

    pub fn rmatmul(
        &self,
        pk: &PK,
        other: &PlaintextVector,
        lshape: Vec<usize>,
        rshape: Vec<usize>,
    ) -> CiphertextVector {
        // rshape, lshape -> rshape[0] x lshape[1]
        // other, self
        // 4 x 2, 2 x 5
        // ik, kj  -> ij
        let mut data = vec![Ciphertext::zero(); lshape[1] * rshape[0]];
        for i in 0..rshape[0] {
            // 4
            for j in 0..lshape[1] {
                // 5
                for k in 0..rshape[1] {
                    // 2
                    data[i * lshape[1] + j].add_assign(
                        &self.data[k * lshape[1] + j].mul(&other.data[i * rshape[1] + k], &pk),
                        &pk,
                    );
                }
            }
        }
        CiphertextVector { data }
    }
}

impl PlaintextVector {
    pub fn get_stride(&mut self, index: usize, stride: usize) -> PlaintextVector {
        let start = index * stride;
        let end = start + stride;
        let data = self.data[start..end].to_vec();
        PlaintextVector { data }
    }
    pub fn tolist(&self) -> Vec<Plaintext> {
        self.data
            .iter()
            .map(|x| x.clone())
            .collect()
    }
}