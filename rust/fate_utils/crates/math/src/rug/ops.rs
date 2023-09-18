use super::BInt;
use core::ops::{Add, Div, Mul, Rem, Shl, Shr, Sub};
use rug::Integer;

/// macro for implementitions of Add, Sub, Mul, Div
macro_rules! arith_bint_bint {
    (
        $( $Imp:ident { $method:ident } ),*
    ) => (
        $(
            // BInt # BInt -> BInt
            impl $Imp<BInt> for BInt {
                type Output = BInt;
                #[inline]
                fn $method(self, rhs: BInt) -> BInt {
                    BInt($Imp::$method(self.0, rhs.0))
                }
            }
            // BInt # &BInt -> BInt
            impl $Imp<BInt> for &BInt {
                type Output = BInt;
                #[inline]
                fn $method(self, rhs: BInt) -> BInt {
                    BInt($Imp::$method(&self.0, rhs.0))
                }
            }
            // &BInt # BInt -> BInt
            impl $Imp<&BInt> for BInt {
                type Output = BInt;
                #[inline]
                fn $method(self, rhs: &BInt) -> BInt {
                    BInt($Imp::$method(self.0, &rhs.0))
                }
            }
            // &BInt # &BInt -> BInt
            impl $Imp<&BInt> for &BInt {
                type Output = BInt;
                #[inline]
                fn $method(self, rhs: &BInt) -> BInt {
                    BInt(Integer::from($Imp::$method(&self.0, &rhs.0)))
                }
            }
        )*
    );
}

macro_rules! arith_bint_primint {
    (
        $( $Imp:ident { $method:ident } ),*;
        $primint:ident
    ) => {
        $(
            impl $Imp<$primint> for BInt{
                type Output = BInt;
                #[inline]
                fn $method(self, rhs: $primint) -> Self::Output {
                    BInt($Imp::$method(self.0, rhs))
                }
            }
            impl $Imp<$primint> for &BInt{
                type Output = BInt;
                #[inline]
                fn $method(self, rhs: $primint) -> Self::Output {
                    BInt(Integer::from($Imp::$method(&self.0, rhs)))
                }
            }
        )*
    }
}

macro_rules! arith_bint_primint_all{
    ( $( $primint:ident ),* ) => {
        $(
            arith_bint_primint!{
                Add {add}, Sub {sub}, Mul {mul}, Div {div};
                $primint
            }
            impl From<$primint> for BInt {
                fn from(a: $primint) -> Self {
                    BInt(Integer::from(a))
                }
            }
        )*
    }
}

// pub(crate) use arith_bint_bint;
// pub(crate) use arith_bint_primint;
// pub(crate) use arith_bint_primint_all;

arith_bint_bint!(
    Add { add },
    Sub { sub },
    Mul { mul },
    Div { div },
    Rem { rem }
);
arith_bint_primint_all!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

impl Mul<rug::Float> for BInt {
    type Output = rug::Float;
    fn mul(self, rhs: rug::Float) -> Self::Output {
        self.0 * rhs
    }
}

macro_rules! impl_sh {
    ( $($int_type: ty),* ) => {
        $(
            impl Shl<$int_type> for BInt {
                type Output = BInt;
                fn shl(self, rhs: $int_type) -> Self::Output {
                    BInt(self.0 << rhs)
                }
            }
            impl Shr<$int_type> for BInt {
                type Output = BInt;
                fn shr(self, rhs: $int_type) -> Self::Output {
                    BInt(self.0 >> rhs)
                }
            }
        )*
    }
}
impl_sh!(i32, u32);
