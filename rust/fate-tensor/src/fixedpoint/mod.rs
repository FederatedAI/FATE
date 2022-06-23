mod coder;
mod frexp;
use crate::math::BInt;
use crate::paillier;
pub(crate) use coder::CouldCode;
pub use coder::FixedpointCoder;
use serde::{Deserialize, Serialize};

const BASE: u32 = 16;

/// fixedpoint plaintext
#[derive(Debug)]
pub struct PT {
    pub significant: paillier::PT,
    pub exp: i32,
}

/// fixedpoint ciphertext
/// raw paillier ciphertext represents encryped significant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CT {
    significant_encryped: paillier::CT,
    exp: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PK {
    pub pk: paillier::PK,
    pub coder: coder::FixedpointCoder,
}

impl PK {
    fn new(pk: paillier::PK) -> PK {
        let coder = coder::FixedpointCoder::new(&pk.n);
        PK { pk, coder }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SK {
    pub sk: paillier::SK,
    pub coder: coder::FixedpointCoder,
}

impl SK {
    fn new(sk: paillier::SK) -> SK {
        let coder = coder::FixedpointCoder::new(&sk.n);
        SK { sk, coder }
    }
}

pub fn keygen(bit_lenght: u32) -> (SK, PK) {
    let (sk, pk) = paillier::keygen(bit_lenght);
    (SK::new(sk), PK::new(pk))
}

impl PK {
    pub fn encrypt(&self, plaintext: &PT) -> CT {
        let exp = plaintext.exp;
        let encode = self.pk.encrypt(&plaintext.significant, true);
        CT {
            significant_encryped: encode,
            exp,
        }
    }
}

impl SK {
    pub fn decrypt(&self, ciphertext: &CT) -> PT {
        let exp = ciphertext.exp;
        PT {
            significant: self.sk.decrypt(&ciphertext.significant_encryped),
            exp,
        }
    }
}

impl CT {
    fn decrese_exp_to(&self, exp: i32, pk: &paillier::PK) -> CT {
        assert!(exp < self.exp);
        let factor = BInt::from(BASE).pow((self.exp - exp) as u32);
        let significant_encryped = self.significant_encryped.mul_pt(&paillier::PT(factor), pk);
        CT {
            significant_encryped,
            exp,
        }
    }
    pub fn neg(&self, pk: &PK) -> CT {
        self.mul(
            &PT {
                significant: paillier::PT(&pk.pk.n - 1),
                exp: 0,
            },
            pk,
        )
    }
    pub fn add_pt(&self, b: &PT, pk: &PK) -> CT {
        let b = pk.encrypt(b);
        self.add(&b, pk)
    }
    pub fn add(&self, b: &CT, pk: &PK) -> CT {
        let a = self;
        if a.exp > b.exp {
            let a = &a.decrese_exp_to(b.exp, &pk.pk);
            CT {
                significant_encryped: a
                    .significant_encryped
                    .add_ct(&b.significant_encryped, &pk.pk),
                exp: b.exp,
            }
        } else if a.exp < b.exp {
            let b = &b.decrese_exp_to(a.exp, &pk.pk);
            CT {
                significant_encryped: a
                    .significant_encryped
                    .add_ct(&b.significant_encryped, &pk.pk),
                exp: a.exp,
            }
        } else {
            CT {
                significant_encryped: a
                    .significant_encryped
                    .add_ct(&b.significant_encryped, &pk.pk),
                exp: a.exp,
            }
        }
    }
    pub fn mul(&self, b: &PT, pk: &PK) -> CT {
        let encode =
            paillier::CT((&self.significant_encryped.0).pow_mod_ref(&b.significant.0, &pk.pk.ns));
        CT {
            significant_encryped: encode,
            exp: self.exp + b.exp,
        }
    }
}
