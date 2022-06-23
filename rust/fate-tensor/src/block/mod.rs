use super::fixedpoint;
use super::fixedpoint::CouldCode;
use ndarray::{ArrayD, ArrayViewD};
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
}

impl fixedpoint::PK {
    pub fn encrypt_array<T>(&self, array: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        let shape = array.shape().to_vec();
        let data: Vec<fixedpoint::CT> = array
            .iter()
            .map(|e| self.encrypt(&e.encode(&self.coder)))
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
    pub fn add_cipherblock_par(&self, rhs: &Cipherblock) -> Cipherblock {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.pubkey, rhs.pubkey);
        let data: Vec<fixedpoint::CT> = self
            .data
            .par_iter()
            .zip(rhs.data.par_iter())
            .map(|(l, r)| l.add(r, &self.pubkey))
            .collect();
        Cipherblock {
            pubkey: self.pubkey.clone(),
            data,
            shape: self.shape.clone(),
        }
    }
    pub fn add_plaintext_par<T>(&self, rhs: ArrayViewD<T>) -> Cipherblock
    where
        T: CouldCode,
    {
        assert_eq!(self.shape, rhs.shape().to_vec());
        let rhs = self.pubkey.encrypt_array(rhs);
        self.add_cipherblock_par(&rhs)
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
            .map(|e| self.encrypt(&e.encode(&self.coder)))
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
