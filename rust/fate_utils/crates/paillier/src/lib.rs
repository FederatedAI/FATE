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
    pub fn obfuscate(self, pk: &PK) -> CT {
        let rn = pk.random_rn();
        CT(self.0 * rn % &pk.ns)
    }
    pub fn add_ct(&self, ct: &CT, pk: &PK) -> CT {
        CT(&self.0 * &ct.0 % &pk.ns)
    }
    pub fn i_double(&mut self, pk: &PK) {
        self.0.pow_mod_mut(&BInt::from(2), &pk.ns);
    }
    pub fn mul_pt(&self, b: &PT, pk: &PK) -> CT {
        CT(self.0.pow_mod_ref(&b.0, &pk.ns))
    }
}

#[derive(Default, Clone, Deserialize, Serialize, Debug, PartialEq)]
pub struct PT(pub BInt); // plaintest

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PK {
    pub n: BInt,
    pub ns: BInt, // n * n
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SK {
    p: BInt,
    q: BInt,
    pub n: BInt,
    // n = p * q
    p_minus_one: BInt,
    q_minus_one: BInt,
    ps: BInt,
    qs: BInt,
    p_invert: BInt,
    // p^{-1} mod q
    hp: BInt,
    hq: BInt,
}

/// generate paillier keypairs with providing bit lenght
pub fn keygen(bit_lenght: u32) -> (SK, PK) {
    assert_eq!(bit_lenght % 2, 0);
    let prime_bit_size = bit_lenght / 2;
    // generate prime p and q such that num_bit(p) * num_bit(q) == bit_length
    // and p != q
    let (mut p, mut q, mut n): (BInt, BInt, BInt);
    loop {
        p = BInt::gen_prime(prime_bit_size);
        q = BInt::gen_prime(prime_bit_size);
        n = &p * &q;
        if p != q && n.significant_bits() == bit_lenght {
            break;
        }
    }
    (SK::new(p, q, n.clone()), PK::new(n))
}

impl PK {
    fn new(n: BInt) -> PK {
        let ns = &n * &n;
        PK { n, ns }
    }
    fn random_rn(&self) -> BInt {
        let mut r = BInt::gen_positive_integer(&self.n);
        r.pow_mod_mut(&self.n, &self.ns);
        r
    }
    /// encrypt plaintext
    ///
    /// ```math
    /// (plaintext \cdot n + 1)r^n \pmod{n^2}
    /// ```
    pub fn encrypt(&self, plaintext: &PT, obfuscate: bool) -> CT {
        let nude_ciphertext = {
            if plaintext.0 > self.n.clone() >> 2u32 {
                let neg_plaintext = &self.n - &plaintext.0;
                let neg_ciphertext = (&self.n * neg_plaintext + ONE) % &self.ns;
                neg_ciphertext.invert(&self.ns)
            } else {
                (&plaintext.0 * &self.n + 1) % &self.ns
            }
        };
        let e = if obfuscate {
            let rn = self.random_rn();
            nude_ciphertext * rn % &self.ns
        } else {
            nude_ciphertext
        };
        CT(e)
    }
}

impl SK {
    fn new(p: BInt, q: BInt, n: BInt) -> SK {
        assert!(p != q, "p == q");
        let (p, q) = if p < q { (p, q) } else { (q, p) }; // p < q
        let ps = &p * &p; // p * p
        let qs = &q * &q; // q * q
        let p_invert = p.invert_ref(&q); // p^{-1} mod q
        let g = &p * &q + ONE; // g = p * q + 1
        let p_minus_one = &p - ONE;
        let q_minus_one = &q - ONE;
        // (((g^{p-1} mod p^2) - 1) / p)^{-1}
        let hp = ((g.pow_mod_ref(&p_minus_one, &ps) - ONE) / &p).invert(&p);
        // (((g^{q-1} mod q^2) - 1) / q)^{-1}
        let hq = ((g.pow_mod_ref(&q_minus_one, &qs) - ONE) / &q).invert(&q);
        SK {
            p,
            q,
            n,
            p_minus_one,
            q_minus_one,
            ps,
            qs,
            p_invert,
            hp,
            hq,
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
        let dp = SK::h_function(&c.0, &self.p, &self.p_minus_one, &self.ps, &self.hp);
        let dq = SK::h_function(&c.0, &self.q, &self.q_minus_one, &self.qs, &self.hq);
        let mut o = (((dq - &dp) * &self.p_invert) % &self.q) * &self.p + &dp;
        // TODO: any better way to do this?
        if o < BInt::from(0) {
            o.0.add_assign(&self.n.0)
        }
        PT(o)
    }
    #[inline]
    fn h_function(c: &BInt, p: &BInt, p_1: &BInt, ps: &BInt, hp: &BInt) -> BInt {
        ((c.pow_mod_ref(p_1, ps) - ONE) / p * hp) % p
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
fn decrypt() {
    let (private, public) = keygen(1024);
    let plaintext = PT(BInt::from(25519u32));
    let ciphertext = public.encrypt(&plaintext, true);
    let decrypted = private.decrypt(&ciphertext);
    assert_eq!(plaintext, decrypted)
}
