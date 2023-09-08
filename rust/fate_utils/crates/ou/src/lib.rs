use math::{BInt, ONE};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::ops::AddAssign;

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
pub struct CT(pub BInt); //ciphertext

impl Display for CT {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CT")
    }
}

impl Default for CT {
    fn default() -> Self {
        todo!()
    }
}

impl<'b> AddAssign<&'b CT> for CT {
    fn add_assign(&mut self, _rhs: &'b CT) {
        todo!()
    }
}

impl CT {
    pub fn zero() -> CT {
        CT(BInt::from(ONE))
    }
    pub fn add_ct(&self, ct: &CT, pk: &PK) -> CT {
        CT(&self.0 * &ct.0 % &pk.n)
    }
    pub fn i_double(&mut self, pk: &PK) {
        self.0.pow_mod_mut(&BInt::from(2), &pk.n);
    }
    pub fn mul_pt(&self, b: &PT, pk: &PK) -> CT {
        CT(self.0.pow_mod_ref(&b.0, &pk.n))
    }
}

#[derive(Default, Clone, Deserialize, Serialize, Debug, PartialEq)]
pub struct PT(pub BInt); // plaintest

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PK {
    pub n: BInt,
    pub g: BInt,
    pub h: BInt,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SK {
    pub p: BInt,
    pub q: BInt,
    pub g: BInt,
    // pub n: BInt,
    // // n = p * q
    // p_minus_one: BInt,
    // q_minus_one: BInt,
    // ps: BInt,
    // qs: BInt,
    // p_invert: BInt,
    // // p^{-1} mod q
    // hp: BInt,
    // hq: BInt,
}

/// generate paillier keypairs with providing bit lenght
pub fn keygen(bit_lenght: u32) -> (SK, PK) {
    let prime_bit_size = bit_lenght / 3;
    let (mut p, mut q, mut n, mut g): (BInt, BInt, BInt, BInt);
    loop {
        p = BInt::gen_prime(prime_bit_size);
        q = BInt::gen_prime(bit_lenght - 2 * prime_bit_size);
        n = &p * &p * &q;
        if p != q && n.significant_bits() == bit_lenght {
            break;
        }
    }
    let p2 = &p * &p;
    let p_1 = &p - 1;
    let n_1 = &n - 1;
    loop {
        g = BInt::gen_positive_integer(&n_1) + 1;
        if g.pow_mod_ref(&p_1, &p2).ne(&BInt::from(1u8)) {
            break;
        }
    }
    let h = g.pow_mod_ref(&n, &n);
    (SK::new(p, q, g.clone()), PK::new(n, g, h))
}

impl PK {
    fn new(n: BInt, g: BInt, h: BInt) -> PK {
        PK { n, g, h }
    }
    /// encrypt plaintext
    ///
    /// ```math
    /// (plaintext \cdot n + 1)r^n \pmod{n^2}
    /// ```
    pub fn encrypt(&self, plaintext: &PT, obfuscate: bool) -> CT {
        let r = BInt::gen_positive_integer(&self.n);
        let c = self.g.pow_mod_ref(&plaintext.0, &self.n) * self.h.pow_mod_ref(&r, &self.n);
        CT(c)
    }
}

impl SK {
    fn new(p: BInt, q: BInt, g: BInt) -> SK {
        assert!(p != q, "p == q");
        SK {
            p,
            q,
            g,
        }
    }
    /// decrypt ciphertext
    ///
    /// crt optimization applied:
    /// ```math
    /// dp = \frac{(c^{p-1} \pmod{p^2})-1}{p}\cdot hp \pmod{p}\\
    /// ```
    /// ```math
    /// dq = \frac{(c^{q-1} \pmod{q^2})-1}{q}\cdot hq \pmod{q}\\
    /// ```
    /// ```math
    /// ((dq - dp)(p^{-1} \pmod{q}) \pmod{q})p + dp
    /// ```
    pub fn decrypt(&self, c: &CT) -> PT {
        let ps = &self.p * &self.p;
        let p_1 = &self.p - 1;
        let dp = SK::h_function(&c.0, &self.p, &p_1, &ps);
        let dq = SK::h_function(&self.g, &self.p, &p_1, &ps);
        let dp1 = dq.invert(&self.p);
        let mut m = (dp * dp1) % &self.p;
        // TODO: any better way to do this?
        if m < BInt::from(0) {
            m.0.add_assign(&self.p.0)
        }

        PT(m)
    }
    #[inline]
    fn h_function(c: &BInt, p: &BInt, p_1: &BInt, ps: &BInt) -> BInt {
        let x = c.pow_mod_ref(p_1, ps) - ONE;
        (x / p) % p
    }
}

#[test]
fn keygen_even_size() {
    keygen(1024);
}

#[test]
#[should_panic]
fn keygen_odd_size() {
    keygen(1023);
}

#[test]
fn test_decrypt() {
    let (private, public) = keygen(1024);
    let plaintext = PT(BInt::from(25519u32));
    let ciphertext = public.encrypt(&plaintext, true);
    let decrypted = private.decrypt(&ciphertext);
    assert_eq!(plaintext, decrypted)
}
#[test]
fn test_add() {
    let (private, public) = keygen(1024);
    let plaintext1 = PT(BInt::from(25519u32));
    let plaintext2 = PT(BInt::from(12345u32));
    let ciphertext1 = public.encrypt(&plaintext1, true);
    let ciphertext2 = public.encrypt(&plaintext2, true);
    let ciphertext3 = ciphertext1.add_ct(&ciphertext2, &public);
    let decrypted = private.decrypt(&ciphertext3);
    assert_eq!(PT(BInt::from(25519u32 + 12345u32)), decrypted)
}
