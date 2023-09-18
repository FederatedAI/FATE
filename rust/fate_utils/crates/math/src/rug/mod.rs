mod ops;
mod random;
// mod serde;
use core::cmp::{PartialEq, PartialOrd};
use rug::Integer;
use rug::{self, ops::Pow};
use serde::{Serialize, Deserialize};


/// newtype of rug::Integer
#[derive(Default, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Debug, Serialize, Deserialize)]
pub struct BInt(pub Integer);
pub const ONE: u8 = 1u8;

impl BInt {
    pub fn from_str_radix(src: &str, radix: i32) -> BInt {
        BInt(Integer::from_str_radix(src, radix).unwrap())
    }
    pub fn significant_bits(&self) -> u32 {
        self.0.significant_bits()
    }
    pub fn pow_mod_mut(&mut self, exp: &BInt, modulo: &BInt) {
        self.0.pow_mod_mut(&exp.0, &modulo.0).unwrap();
    }
    pub fn pow_mod_ref(&self, exp: &BInt, modulo: &BInt) -> BInt {
        BInt(Integer::from(
            self.0.pow_mod_ref(&exp.0, &modulo.0).unwrap(),
        ))
    }
    pub fn invert(self, modulo: &BInt) -> BInt {
        BInt(self.0.invert(&modulo.0).unwrap())
    }
    pub fn invert_ref(&self, modulo: &BInt) -> BInt {
        BInt(Integer::from(self.0.invert_ref(&modulo.0).unwrap()))
    }
    pub fn abs(self) -> BInt {
        BInt(self.0.abs())
    }
    pub fn abs_ref(&self) -> BInt {
        BInt(Integer::from((&self.0).abs_ref()))
    }
    pub fn pow(self, exp: u32) -> BInt {
        BInt(self.0.pow(exp))
    }
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }
    pub fn to_i64(&self) -> i64 {
        self.0.to_i64().expect("cant't convert to i64")
    }
    pub fn to_i128(&self) -> i128 {
        self.0.to_i128().expect("cant't convert to i128")
    }
}
