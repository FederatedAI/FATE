use math::BInt;

fn encrypt() {
    let (_sk, pk) = paillier::keygen(1024);
    let plaintext = paillier::PT(BInt::from_str_radix("1234567890987654321", 10));
    pk.encrypt(&plaintext, true);
}

iai::main!(encrypt);
