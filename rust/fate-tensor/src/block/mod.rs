use std::ops::Index;

use super::fixedpoint;
use super::fixedpoint::CouldCode;
use ndarray::{ArrayD, ArrayViewD};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
mod matmul;

#[derive(Clone, Serialize, Deserialize)]
pub struct Cipherblock {
    pub pk: fixedpoint::PK,
    pub data: Vec<fixedpoint::CT>,
    pub shape: Vec<usize>,
}

impl Index<(usize, usize)> for Cipherblock {
    type Output = fixedpoint::CT;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.shape[1] + index.1]
    }
}
impl Cipherblock {
    pub fn map<F>(&self, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT) -> fixedpoint::CT,
    {
        Cipherblock {
            pk: self.pk.clone(),
            data: self.data.iter().map(func).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn ops_cb_cb<F>(lhs: &Cipherblock, rhs: &Cipherblock, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT, &fixedpoint::CT, &fixedpoint::PK) -> fixedpoint::CT,
    {
        assert_eq!(lhs.shape, rhs.shape);
        assert_eq!(lhs.pk, rhs.pk);
        let lhs_iter = lhs.data.iter();
        let rhs_iter = rhs.data.iter();
        let data: Vec<fixedpoint::CT> = lhs_iter
            .zip(rhs_iter)
            .map(|(l, r)| func(l, r, &lhs.pk))
            .collect();
        Cipherblock {
            pk: lhs.pk.clone(),
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
            .map(|(l, r)| func(l, &r.encode(&lhs.pk.coder), &lhs.pk))
            .collect();
        Cipherblock {
            pk: lhs.pk.clone(),
            data,
            shape: lhs.shape.clone(),
        }
    }
    pub fn neg(&self) -> Cipherblock {
        self.map(|x| x.neg(&self.pk))
    }
    pub fn add_cipherblock(&self, rhs: &Cipherblock) -> Cipherblock {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.pk, rhs.pk);
        let data: Vec<fixedpoint::CT> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(l, r)| l.add(r, &self.pk))
            .collect();
        Cipherblock {
            pk: self.pk.clone(),
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
        let rhs = self.pk.encrypt_array(rhs);
        self.add_cipherblock(&rhs)
    }
    pub fn sub_plaintext<T>(&self, rhs: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        assert_eq!(self.shape, rhs.shape().to_vec());
        let rhs = self.pk.encrypt_array(rhs);
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
            .map(|(l, r)| l.mul(&r.encode(&self.pk.coder), &self.pk))
            .collect();
        Cipherblock {
            pk: self.pk.clone(),
            data,
            shape: self.shape.clone(),
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
            pk: self.clone(),
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
        assert_eq!(lhs.pk, rhs.pk);
        let lhs_iter = lhs.data.par_iter();
        let rhs_iter = rhs.data.par_iter();
        let data: Vec<fixedpoint::CT> = lhs_iter
            .zip(rhs_iter)
            .map(|(l, r)| func(l, r, &lhs.pk))
            .collect();
        Cipherblock {
            pk: lhs.pk.clone(),
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
            .map(|(l, r)| func(l, &r.encode(&lhs.pk.coder), &lhs.pk))
            .collect();
        Cipherblock {
            pk: lhs.pk.clone(),
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
            pk: self.clone(),
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
