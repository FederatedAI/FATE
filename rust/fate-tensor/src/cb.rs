use super::{block, fixedpoint, fixedpoint::CouldCode, Cipherblock, PK, SK};
use ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayViewD};

fn binary_plain<F, T>(this: &Cipherblock, other: ArrayViewD<T>, func: F) -> Cipherblock
where
    F: Fn(&block::Cipherblock, ArrayViewD<T>) -> block::Cipherblock,
{
    Cipherblock::new(func(this.unwrap(), other))
}

fn binary_cipher<F>(this: &Cipherblock, other: &Cipherblock, func: F) -> Cipherblock
where
    F: Fn(&block::Cipherblock, &block::Cipherblock) -> block::Cipherblock,
{
    let a = this.unwrap();
    let b = other.unwrap();
    Cipherblock::new(func(a, b))
}

macro_rules! impl_ops_cipher {
    ($name:ident,$fn:expr) => {
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            binary_cipher(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_cb(lhs, rhs, $fn)
            })
        }
    };
    ($name:ident,$fn:expr,$feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            binary_cipher(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_cb_par(lhs, rhs, $fn)
            })
        }
    };
}
macro_rules! impl_ops_plain {
    ($name:ident,$fn:expr) => {
        pub fn $name<T>(&self, other: ArrayViewD<T>) -> Cipherblock
        where
            T: fixedpoint::CouldCode,
        {
            binary_plain(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_pt(lhs, rhs, $fn)
            })
        }
    };
    ($name:ident,$fn:expr,$feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name<T>(&self, other: ArrayViewD<T>) -> Cipherblock
        where
            T: fixedpoint::CouldCode + Sync + Send,
        {
            binary_plain(self, other, |lhs, rhs| {
                block::Cipherblock::ops_cb_pt_par(lhs, rhs, $fn)
            })
        }
    };
}
macro_rules! impl_ops_matmul {
    ($name:ident, $fn:expr, $oty:ident) => {
        pub fn $name<T: CouldCode + Sync>(&self, other: $oty<T>) -> Cipherblock {
            Cipherblock::new($fn(self.unwrap(), other))
        }
    };
    ($name:ident, $fn:expr, $oty:ident, $feature:ident) => {
        #[cfg(feature = "rayon")]
        pub fn $name<T: CouldCode + Sync>(&self, other: $oty<T>) -> Cipherblock {
            Cipherblock::new($fn(self.unwrap(), other))
        }
    };
}
impl Cipherblock {
    fn new(cb: block::Cipherblock) -> Cipherblock {
        Cipherblock(Some(cb))
    }
    fn unwrap(&self) -> &block::Cipherblock {
        self.0.as_ref().unwrap()
    }

    impl_ops_cipher!(add_cb, fixedpoint::CT::add);
    impl_ops_cipher!(sub_cb, fixedpoint::CT::sub);
    impl_ops_plain!(add_plaintext, fixedpoint::CT::add_pt);
    impl_ops_plain!(sub_plaintext, fixedpoint::CT::sub_pt);
    impl_ops_plain!(mul_plaintext, fixedpoint::CT::mul);

    // matmul
    impl_ops_matmul!(
        matmul_plaintext_ix1,
        block::Cipherblock::matmul_plaintext_ix1,
        ArrayView1
    );
    impl_ops_matmul!(
        rmatmul_plaintext_ix1,
        block::Cipherblock::rmatmul_plaintext_ix1,
        ArrayView1
    );
    impl_ops_matmul!(
        matmul_plaintext_ix2,
        block::Cipherblock::matmul_plaintext_ix2,
        ArrayView2
    );
    impl_ops_matmul!(
        rmatmul_plaintext_ix2,
        block::Cipherblock::rmatmul_plaintext_ix2,
        ArrayView2
    );

    //par
    impl_ops_cipher!(add_cb_par, fixedpoint::CT::add, rayon);
    impl_ops_cipher!(sub_cb_par, fixedpoint::CT::sub, rayon);
    impl_ops_plain!(add_plaintext_par, fixedpoint::CT::add_pt, rayon);
    impl_ops_plain!(sub_plaintext_par, fixedpoint::CT::sub_pt, rayon);
    impl_ops_plain!(mul_plaintext_par, fixedpoint::CT::mul, rayon);

    // matmul
    impl_ops_matmul!(
        matmul_plaintext_ix1_par,
        block::Cipherblock::matmul_plaintext_ix1_par,
        ArrayView1,
        rayon
    );
    impl_ops_matmul!(
        rmatmul_plaintext_ix1_par,
        block::Cipherblock::rmatmul_plaintext_ix1_par,
        ArrayView1,
        rayon
    );
    impl_ops_matmul!(
        matmul_plaintext_ix2_par,
        block::Cipherblock::matmul_plaintext_ix2_par,
        ArrayView2,
        rayon
    );
    impl_ops_matmul!(
        rmatmul_plaintext_ix2_par,
        block::Cipherblock::rmatmul_plaintext_ix2_par,
        ArrayView2,
        rayon
    );
}

impl SK {
    pub fn decrypt_array<T: CouldCode + numpy::Element>(&self, a: &Cipherblock) -> ArrayD<T> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array(array)
    }
    #[cfg(feature = "rayon")]
    pub fn decrypt_array_par<T: CouldCode + numpy::Element>(&self, a: &Cipherblock) -> ArrayD<T> {
        let array = a.0.as_ref().unwrap();
        self.sk.decrypt_array_par(array)
    }
}

impl PK {
    pub fn encrypt_array<T: CouldCode>(&self, array: ArrayViewD<T>) -> Cipherblock {
        Cipherblock::new(self.pk.encrypt_array(array))
    }
    #[cfg(feature = "rayon")]
    pub fn encrypt_array_par<T: CouldCode + Sync + Send>(
        &self,
        array: ArrayViewD<T>,
    ) -> Cipherblock {
        Cipherblock::new(self.pk.encrypt_array_par(array))
    }
}
