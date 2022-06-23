use rug::Integer;
use rust_tensor::paillier::keygen;

fn encrypt() {
    let (prikey, pubkey) = keygen(1024);
    let plaintext = Integer::from_str_radix("1234567890987654321", 10).unwrap();
    pubkey.encrypt(&plaintext);
}

iai::main!(encrypt);
