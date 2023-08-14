use std::ops::{AddAssign, ShlAssign, SubAssign};
use super::frexp::Frexp;
use super::PT;
use crate::paillier;
use math::BInt;
use rug::{self, Integer, ops::Pow};
use serde::{Deserialize, Serialize};

const FLOAT_MANTISSA_BITS: u32 = 53;
const LOG2_BASE: u32 = 4;
const BASE: u32 = 16;
const MAX_INT_FRACTION: u8 = 2;

/// fixedpoint encoder
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FixedpointCoder {
    pub n: BInt,
    pub max_int: BInt,
}

impl FixedpointCoder {
    pub fn new(n: &BInt) -> Self {
        FixedpointCoder {
            n: n.clone(),
            max_int: n / MAX_INT_FRACTION,
        }
    }
    pub fn encode_i64(&self, plaintext: i64) -> PT {
        let significant = paillier::PT(if plaintext < 0 {
            BInt::from(&self.n + plaintext)
        } else {
            BInt::from(plaintext)
        });
        PT {
            significant,
            exp: 0,
        }
    }
    pub fn pack(&self, plaintexts: &[u64], num_shift_bit: usize) -> PT {
        let significant = plaintexts.iter().fold(Integer::default(), |mut x, v| {
            x.shl_assign(num_shift_bit);
            x.add_assign(v);
            x
        });
        PT {
            significant: paillier::PT(BInt(significant)),
            exp: 0,
        }
    }
    pub fn unpack(&self, encoded: &PT, num_shift_bit: usize, num: usize) -> Vec<u64> {
        let mut significant = encoded.significant.0.0.clone();
        let mut mask = Integer::from(1u64 << num_shift_bit);
        mask.sub_assign(1);

        let mut result = Vec::with_capacity(num);
        for _ in 0..num {
            let value = Integer::from(significant.clone() & mask.clone()).to_u64().unwrap();
            result.push(value);
            significant >>= num_shift_bit;
        }
        result.reverse();
        result
    }
    pub fn encode_i32(&self, plaintext: i32) -> PT {
        let significant = paillier::PT(if plaintext < 0 {
            BInt::from(&self.n + plaintext)
        } else {
            BInt::from(plaintext)
        });
        PT {
            significant,
            exp: 0,
        }
    }
    pub fn decode_i64(&self, encoded: &PT) -> i64 {
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
    pub fn decode_i32(&self, encoded: &PT) -> i32 {
        // Todo: could be improved
        self.decode_f64(encoded) as i32
    }

    pub fn encode_f64(&self, plaintext: f64) -> PT {
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
        PT {
            significant: paillier::PT(significant),
            exp,
        }
    }
    pub fn decode_f64(&self, encoded: &PT) -> f64 {
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
    pub fn encode_f32(&self, plaintext: f32) -> PT {
        self.encode_f64(plaintext as f64)
    }
    pub fn decode_f32(&self, encoded: &PT) -> f32 {
        self.decode_f64(encoded) as f32
    }
}

pub trait CouldCode {
    fn encode(&self, coder: &FixedpointCoder) -> PT;
    fn decode(pt: &PT, coder: &FixedpointCoder) -> Self;
}

impl CouldCode for f64 {
    fn encode(&self, coder: &FixedpointCoder) -> PT {
        coder.encode_f64(*self)
    }
    fn decode(pt: &PT, coder: &FixedpointCoder) -> Self {
        coder.decode_f64(pt)
    }
}

impl CouldCode for i64 {
    fn encode(&self, coder: &FixedpointCoder) -> PT {
        coder.encode_i64(*self)
    }
    fn decode(pt: &PT, coder: &FixedpointCoder) -> Self {
        coder.decode_i64(pt)
    }
}

impl CouldCode for i32 {
    fn encode(&self, coder: &FixedpointCoder) -> PT {
        coder.encode_i32(*self)
    }
    fn decode(pt: &PT, coder: &FixedpointCoder) -> Self {
        coder.decode_i32(pt)
    }
}

impl CouldCode for f32 {
    fn encode(&self, coder: &FixedpointCoder) -> PT {
        coder.encode_f32(*self)
    }
    fn decode(pt: &PT, coder: &FixedpointCoder) -> Self {
        coder.decode_f32(pt)
    }
}
