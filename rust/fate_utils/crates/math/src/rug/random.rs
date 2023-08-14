use super::BInt;
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;
use rug::rand::{RandGen, RandState};
use rug::Integer;
pub(crate) struct StdRngGen(StdRng);
impl RandGen for StdRngGen {
    fn gen(&mut self) -> u32 {
        self.0.next_u32()
    }
}

impl StdRngGen {
    pub fn new() -> StdRngGen {
        StdRngGen(StdRng::from_entropy())
    }
    fn rand_state(&mut self) -> RandState {
        RandState::new_custom(self)
    }

    fn gen_positive_integer(&mut self, bound: &BInt) -> BInt {
        BInt(Integer::from(&bound.0 - 1).random_below(&mut self.rand_state()) + 1u8)
    }

    /// generate random prime about `bit_size` bit lenght
    pub fn gen_prime(&mut self, bit_size: u32) -> BInt {
        let mut prime = Integer::from(Integer::random_bits(bit_size, &mut self.rand_state()));
        prime.set_bit(bit_size - 1, true);
        prime.next_prime_mut();
        BInt(prime)
    }
}

impl BInt {
    pub fn gen_positive_integer(bound: &BInt) -> BInt {
        StdRngGen::new().gen_positive_integer(bound)
    }

    /// generate random prime about `bit_size` bit lenght
    pub fn gen_prime(bit_size: u32) -> BInt {
        StdRngGen::new().gen_prime(bit_size)
    }
}
