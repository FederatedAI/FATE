use super::{Cipherblock, PK, SK};
use fixedpoint;
use fixedpoint::CouldCode;
use ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayViewD};
use paillier_tensor::block;

fn operation_with_arrayview_dyn<F, T>(
    this: &Cipherblock,
    other: ArrayViewD<T>,
    func: F,
) -> Cipherblock
where
    F: Fn(&block::Cipherblock, ArrayViewD<T>) -> block::Cipherblock,
{
    Cipherblock::new(func(this.unwrap(), other))
}

fn operation_with_cipherblock<F>(this: &Cipherblock, other: &Cipherblock, func: F) -> Cipherblock
where
    F: Fn(&block::Cipherblock, &block::Cipherblock) -> block::Cipherblock,
{
    let a = this.unwrap();
    let b = other.unwrap();
    Cipherblock::new(func(a, b))
}

fn operation_with_scalar<F, T>(this: &Cipherblock, other: T, func: F) -> Cipherblock
where
    F: Fn(&block::Cipherblock, T) -> block::Cipherblock,
{
    Cipherblock::new(func(this.unwrap(), other))
}

macro_rules! impl_ops_cipher_scalar {
    ($name:ident,$fn:expr) => {
        pub fn $name(&self, other: &fixedpoint::CT) -> Cipherblock {
            operation_with_scalar(self, other, |lhs, rhs| {
                block::Cipherblock::map(lhs, |c| $fn(c, rhs, &lhs.pk))
            })
        }
    };
}
macro_rules! impl_ops_plaintext_scalar {
    ($name:ident,$fn:expr) => {
        pub fn $name<T>(&self, other: T) -> Cipherblock
        where
            T: CouldCode,
        {
            operation_with_scalar(self, other, |lhs, rhs| {
                block::Cipherblock::map(lhs, |c| $fn(c, &rhs.encode(&lhs.pk.coder), &lhs.pk))
            })
        }
    };
}
macro_rules! impl_ops_cipher {
    ($name:ident,$fn:expr) => {
        pub fn $name(&self, other: &Cipherblock) -> Cipherblock {
            operation_with_cipherblock(self, other, |lhs, rhs| {
                block::Cipherblock::binary_cipherblock_cipherblock(lhs, rhs, $fn)
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
            operation_with_arrayview_dyn(self, other, |lhs, rhs| {
                block::Cipherblock::binary_cipherblock_plaintext(lhs, rhs, $fn)
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
}
impl Cipherblock {
    fn new(cb: block::Cipherblock) -> Cipherblock {
        Cipherblock(Some(cb))
    }
    fn unwrap(&self) -> &block::Cipherblock {
        self.0.as_ref().unwrap()
    }
    impl_ops_cipher!(add_cb, fixedpoint::CT::add);
    impl_ops_plain!(add_plaintext, fixedpoint::CT::add_pt);
    impl_ops_cipher_scalar!(add_cipher_scalar, fixedpoint::CT::add);
    impl_ops_plaintext_scalar!(add_plaintext_scalar, fixedpoint::CT::add_pt);
    impl_ops_cipher!(sub_cb, fixedpoint::CT::sub);
    impl_ops_cipher!(rsub_cb, fixedpoint::CT::rsub);
    impl_ops_plain!(sub_plaintext, fixedpoint::CT::sub_pt);
    impl_ops_plain!(rsub_plaintext, fixedpoint::CT::rsub_pt);
    impl_ops_cipher_scalar!(sub_cipher_scalar, fixedpoint::CT::sub);
    impl_ops_cipher_scalar!(rsub_cipher_scalar, fixedpoint::CT::rsub);
    impl_ops_plaintext_scalar!(sub_plaintext_scalar, fixedpoint::CT::sub_pt);
    impl_ops_plaintext_scalar!(rsub_plaintext_scalar, fixedpoint::CT::rsub_pt);
    impl_ops_plain!(mul_plaintext, fixedpoint::CT::mul);
    impl_ops_plaintext_scalar!(mul_plaintext_scalar, fixedpoint::CT::mul);

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
}

impl Cipherblock {
    pub fn sum_cb(&self) -> Cipherblock {
        let cb = self.unwrap();
        if cb.data.len() == 0 {
            panic!("sum of empty Cipherblock")
        }
        let mut sum = cb.data[0].clone();
        for i in 1..cb.data.len() {
            sum.add_assign(&cb.data[i], &cb.pk);
        }
        Cipherblock::new(block::Cipherblock {
            pk: cb.pk.clone(),
            data: vec![sum],
            shape: vec![1],
        })
    }
    pub fn mean_cb(&self) -> Cipherblock {
        let n = self.unwrap().data.len();
        self.sum_cb().mul_plaintext_scalar(1.0f64 / (n as f64))
    }
}

impl SK {
    pub fn decrypt_array<T: CouldCode + numpy::Element>(&self, a: &Cipherblock) -> ArrayD<T> {
        let array = a.0.as_ref().unwrap();
        block::decrypt_array(self.as_ref(), array)
    }
}

impl PK {
    pub fn encrypt_array<T: CouldCode>(&self, array: ArrayViewD<T>) -> Cipherblock {
        Cipherblock::new(block::encrypt_array(self.as_ref(), array))
    }
}
