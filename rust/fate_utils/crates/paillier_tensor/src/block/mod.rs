use std::ops::Index;

use fixedpoint;
use fixedpoint::CouldCode;
use ndarray::{ArrayD, ArrayViewD};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod matmul;

#[derive(Clone, Serialize, Deserialize, Debug)]
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
    pub fn reshape(&self, shape: Vec<usize>) -> Cipherblock {
        let s1: usize = self.shape.iter().product();
        let s2: usize = shape.iter().product();
        if s1 != s2 {
            panic!("reshape failed, {} vs {}", s1, s2);
        };
        Cipherblock {
            pk: self.pk.clone(),
            data: self.data.clone(),
            shape,
        }
    }
    pub fn slice0(&self, index: usize) -> Cipherblock {
        let stride: usize = self.shape[1..].iter().product();
        let start = index * stride;
        let end = start + stride;
        Cipherblock {
            pk: self.pk.clone(),
            data: self.data[start..end].to_vec(),
            shape: self.shape[1..].to_vec(),
        }
    }
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
    pub fn agg<F, T>(&self, init: T, f: F) -> T
    where
        F: Fn(T, &fixedpoint::CT) -> T,
    {
        self.data.iter().fold(init, f)
    }
    pub fn binary_cipherblock_cipherblock<F>(
        lhs: &Cipherblock,
        rhs: &Cipherblock,
        func: F,
    ) -> Cipherblock
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
    pub fn binary_cipherblock_plaintext<F, T>(
        lhs: &Cipherblock,
        rhs: ArrayViewD<T>,
        func: F,
    ) -> Cipherblock
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
}

pub fn encrypt_array<T>(pk: &fixedpoint::PK, array: ArrayViewD<T>) -> Cipherblock
where
    T: CouldCode,
{
    let shape = array.shape().to_vec();
    let data: Vec<fixedpoint::CT> = array
        .iter()
        .map(|e| pk.encrypt(&e.encode(&pk.coder), true))
        .collect();
    Cipherblock {
        pk: pk.clone(),
        data,
        shape,
    }
}

pub fn decrypt_array<T>(sk: &fixedpoint::SK, array: &Cipherblock) -> ArrayD<T>
where
    T: CouldCode,
{
    let shape = array.shape.as_slice();
    let data = array
        .data
        .iter()
        .map(|e| T::decode(&sk.decrypt(e), &sk.coder))
        .collect();
    ArrayD::from_shape_vec(shape, data).unwrap()
}

impl Cipherblock {
    pub fn agg_par<F, T, ID, OP>(&self, identity: ID, f: F, op: OP) -> T
    where
        F: Fn(T, &fixedpoint::CT) -> T + Send + Sync,
        ID: Fn() -> T + Send + Sync,
        OP: Fn(T, T) -> T + Send + Sync,
        T: Send,
    {
        self.data
            .par_iter()
            .fold(&identity, f)
            .reduce(&identity, op)
    }
    pub fn map_par<F>(&self, func: F) -> Cipherblock
    where
        F: Fn(&fixedpoint::CT) -> fixedpoint::CT + Sync + Send,
    {
        Cipherblock {
            pk: self.pk.clone(),
            data: self.data.par_iter().map(func).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn binary_cipherblock_cipherblock_par<F>(
        lhs: &Cipherblock,
        rhs: &Cipherblock,
        func: F,
    ) -> Cipherblock
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
    pub fn binary_cipherblock_plaintext_par<F, T>(
        lhs: &Cipherblock,
        rhs: ArrayViewD<T>,
        func: F,
    ) -> Cipherblock
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

// impl fixedpoint::PK {
//     pub fn encrypt_array_par<T>(&self, array: ArrayViewD<T>) -> Cipherblock
//     where
//         T: CouldCode + Send + Sync,
//     {
//         let shape = array.shape().to_vec();
//         let data: Vec<fixedpoint::CT> = array
//             .into_par_iter()
//             .map(|e| self.encrypt(&e.encode(&self.coder), true))
//             .collect();
//         Cipherblock {
//             pk: self.clone(),
//             data,
//             shape,
//         }
//     }
// }

// impl fixedpoint::SK {
//     pub fn decrypt_array_par<T>(&self, array: &Cipherblock) -> ArrayD<T>
//     where
//         T: CouldCode + Send,
//     {
//         let shape = array.shape.as_slice();
//         let data = array
//             .data
//             .par_iter()
//             .map(|e| T::decode(&self.decrypt(e), &self.coder))
//             .collect();
//         ArrayD::from_shape_vec(shape, data).unwrap()
//     }
// }
