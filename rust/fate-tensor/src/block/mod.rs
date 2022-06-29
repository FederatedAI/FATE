use super::fixedpoint;
use super::fixedpoint::CouldCode;
use itertools;
use ndarray::{ArrayD, ArrayView2, ArrayViewD};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Cipherblock {
    pub pubkey: fixedpoint::PK,
    pub data: Vec<fixedpoint::CT>,
    pub shape: Vec<usize>,
}

impl Cipherblock {
    pub fn map<F>(&self, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT) -> fixedpoint::CT,
    {
        Cipherblock {
            pubkey: self.pubkey.clone(),
            data: self.data.iter().map(func).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn ops_cb_cb<F>(lhs: &Cipherblock, rhs: &Cipherblock, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT, &fixedpoint::CT, &fixedpoint::PK) -> fixedpoint::CT,
    {
        assert_eq!(lhs.shape, rhs.shape);
        assert_eq!(lhs.pubkey, rhs.pubkey);
        let lhs_iter = lhs.data.iter();
        let rhs_iter = rhs.data.iter();
        let data: Vec<fixedpoint::CT> = lhs_iter
            .zip(rhs_iter)
            .map(|(l, r)| func(l, r, &lhs.pubkey))
            .collect();
        Cipherblock {
            pubkey: lhs.pubkey.clone(),
            data,
            shape: lhs.shape.clone(),
        }
    }
    pub fn ops_cb_pt<F, T>(lhs: &Cipherblock, rhs: ArrayViewD<T>, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT, &fixedpoint::PT, &fixedpoint::PK) -> fixedpoint::CT,
        T: CouldCode,
    {
        assert_eq!(lhs.shape, rhs.shape().to_vec());
        let lhs_iter = lhs.data.iter();
        let rhs_iter = rhs.iter();
        let data: Vec<fixedpoint::CT> = lhs_iter
            .zip(rhs_iter)
            .map(|(l, r)| func(l, &r.encode(&lhs.pubkey.coder), &lhs.pubkey))
            .collect();
        Cipherblock {
            pubkey: lhs.pubkey.clone(),
            data,
            shape: lhs.shape.clone(),
        }
    }
    pub fn neg(&self) -> Cipherblock {
        self.map(|x| x.neg(&self.pubkey))
    }
    pub fn add_cipherblock(&self, rhs: &Cipherblock) -> Cipherblock {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.pubkey, rhs.pubkey);
        let data: Vec<fixedpoint::CT> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(l, r)| l.add(r, &self.pubkey))
            .collect();
        Cipherblock {
            pubkey: self.pubkey.clone(),
            data,
            shape: self.shape.clone(),
        }
    }
    pub fn sub_cipherblock(&self, rhs: &Cipherblock) -> Cipherblock {
        self.add_cipherblock(&rhs.neg())
    }
    pub fn add_plaintext<T>(&self, rhs: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        assert_eq!(self.shape, rhs.shape().to_vec());
        let rhs = self.pubkey.encrypt_array(rhs);
        self.add_cipherblock(&rhs)
    }
    pub fn sub_plaintext<T>(&self, rhs: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        assert_eq!(self.shape, rhs.shape().to_vec());
        let rhs = self.pubkey.encrypt_array(rhs);
        self.sub_cipherblock(&rhs)
    }
    pub fn mul_plaintext<T>(&self, rhs: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        assert_eq!(self.shape, rhs.shape().to_vec());
        let data: Vec<fixedpoint::CT> = self
            .data
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l.mul(&r.encode(&self.pubkey.coder), &self.pubkey))
            .collect();
        Cipherblock {
            pubkey: self.pubkey.clone(),
            data,
            shape: self.shape.clone(),
        }
    }
    pub fn matmul_plaintext_ix2<T>(&self, rhs: ArrayView2<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        match self.shape.as_slice() {
            &[m, k1] => {
                let (k2, n) = rhs.dim();
                if k1 != k2 || m.checked_mul(n).is_none() {
                    panic!("dot shape error: ({}, {}) x ({}, {})", m, k1, k2, n);
                }
                let mut c: Vec<fixedpoint::CT> = vec![fixedpoint::CT::zero(); m * n];
                let coder = &self.pubkey.coder;
                let pk = &self.pubkey;
                itertools::iproduct!(0..m, 0..n)
                    .zip(c.iter_mut())
                    .for_each(|((i, j), v)| {
                        (0..k1).for_each(|k| {
                            let l = &self.data[i * k1 + k]; // (i, k)
                            let r = rhs[(k, j)].encode(coder); // (k, j)
                            let d = l.mul(&r, &pk); // l * r
                            v.add_assign(&d, &pk); // acc += l * r
                        })
                    });
                Cipherblock {
                    pubkey: pk.clone(),
                    data: c,
                    shape: vec![m, n],
                }
            }
            not_dim2 @ _ => panic!("dot shape error: (?) x {:?}", not_dim2),
        }
    }

    #[cfg(feature = "rayon")]
    pub fn matmul_plaintext_ix2_par<T>(&self, rhs: ArrayView2<T>) -> Cipherblock
    where
        T: CouldCode + Sync,
    {
        match self.shape.as_slice() {
            &[m, k1] => {
                let (k2, n) = rhs.dim();
                if k1 != k2 || m.checked_mul(n).is_none() {
                    panic!("dot shape error: ({}, {}) x ({}, {})", m, k1, k2, n);
                }
                let mut c: Vec<fixedpoint::CT> = vec![fixedpoint::CT::zero(); m * n];
                let coder = &self.pubkey.coder;
                let pk = &self.pubkey;
                let indexes: Vec<(usize, usize)> = itertools::iproduct!(0..m, 0..n).collect();
                indexes
                    .par_iter()
                    .zip(c.par_iter_mut())
                    .for_each(|((i, j), v)| {
                        (0..k1).for_each(|k| {
                            let l = &self.data[i * k1 + k]; // (i, k)
                            let r = rhs[(k, *j)].encode(coder); // (k, j)
                            let d = l.mul(&r, &pk); // l * r
                            v.add_assign(&d, &pk); // acc += l * r
                        });
                    });
                Cipherblock {
                    pubkey: pk.clone(),
                    data: c,
                    shape: vec![m, n],
                }
            }
            not_dim2 @ _ => panic!("dot shape error: (?) x {:?}", not_dim2),
        }
    }
}

impl fixedpoint::PK {
    pub fn encrypt_array<T>(&self, array: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        let shape = array.shape().to_vec();
        let data: Vec<fixedpoint::CT> = array
            .iter()
            .map(|e| self.encrypt(&e.encode(&self.coder), true))
            .collect();
        Cipherblock {
            pubkey: self.clone(),
            data,
            shape,
        }
    }
}

impl fixedpoint::SK {
    pub fn decrypt_array<T>(&self, array: &Cipherblock) -> ArrayD<T>
    where
        T: CouldCode,
    {
        let shape = array.shape.as_slice();
        let data = array
            .data
            .iter()
            .map(|e| T::decode(&self.decrypt(e), &self.coder))
            .collect();
        ArrayD::from_shape_vec(shape, data).unwrap()
    }
}

#[cfg(feature = "rayon")]
impl Cipherblock {
    pub fn ops_cb_cb_par<F>(lhs: &Cipherblock, rhs: &Cipherblock, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT, &fixedpoint::CT, &fixedpoint::PK) -> fixedpoint::CT + Sync,
    {
        assert_eq!(lhs.shape, rhs.shape);
        assert_eq!(lhs.pubkey, rhs.pubkey);
        let lhs_iter = lhs.data.par_iter();
        let rhs_iter = rhs.data.par_iter();
        let data: Vec<fixedpoint::CT> = lhs_iter
            .zip(rhs_iter)
            .map(|(l, r)| func(l, r, &lhs.pubkey))
            .collect();
        Cipherblock {
            pubkey: lhs.pubkey.clone(),
            data,
            shape: lhs.shape.clone(),
        }
    }
    pub fn ops_cb_pt_par<F, T>(lhs: &Cipherblock, rhs: ArrayViewD<T>, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT, &fixedpoint::PT, &fixedpoint::PK) -> fixedpoint::CT + Sync,
        T: CouldCode + Sync + Send,
    {
        assert_eq!(lhs.shape, rhs.shape().to_vec());
        let lhs_iter = lhs.data.par_iter();
        let rhs_iter = rhs.as_slice().unwrap().into_par_iter();
        let data: Vec<fixedpoint::CT> = lhs_iter
            .zip(rhs_iter)
            .map(|(l, r)| func(l, &r.encode(&lhs.pubkey.coder), &lhs.pubkey))
            .collect();
        Cipherblock {
            pubkey: lhs.pubkey.clone(),
            data,
            shape: lhs.shape.clone(),
        }
    }
}

#[cfg(feature = "rayon")]
impl fixedpoint::PK {
    pub fn encrypt_array_par<T>(&self, array: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode + Send + Sync,
    {
        let shape = array.shape().to_vec();
        let data: Vec<fixedpoint::CT> = array
            .into_par_iter()
            .map(|e| self.encrypt(&e.encode(&self.coder), true))
            .collect();
        Cipherblock {
            pubkey: self.clone(),
            data,
            shape,
        }
    }
}

#[cfg(feature = "rayon")]
impl fixedpoint::SK {
    pub fn decrypt_array_par<T>(&self, array: &Cipherblock) -> ArrayD<T>
    where
        T: CouldCode + Send,
    {
        let shape = array.shape.as_slice();
        let data = array
            .data
            .par_iter()
            .map(|e| T::decode(&self.decrypt(e), &self.coder))
            .collect();
        ArrayD::from_shape_vec(shape, data).unwrap()
    }
}
