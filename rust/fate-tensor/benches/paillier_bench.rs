use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fate_tensor::fixedpoint;
use fate_tensor::paillier;
use rug::Integer;
use std::time::Duration;

fn paillier_benchmark(c: &mut Criterion) {
    let (prikey, pubkey) = paillier::keygen(1024);
    let plaintext = Integer::from_str_radix("1234567890987654321", 10).unwrap();
    let ciphertext = pubkey.encrypt(&plaintext);
    let mut group = c.benchmark_group("paillier");
    group.bench_function("encrypt", |b| {
        b.iter(|| black_box(&pubkey).encrypt(black_box(&plaintext)))
    });
    group.bench_function("decrypt", |b| {
        b.iter(|| black_box(&prikey).decrypt(black_box(&ciphertext)))
    });
    let (prikey, pubkey) = fixedpoint::keygen(1024);
    let plaintext = 0.125;
    let ciphertext = pubkey.encrypt(&plaintext);
    group.bench_function("encrypt_fixed", |b| {
        b.iter(|| black_box(&pubkey).encrypt(black_box(&plaintext)))
    });
    group.bench_function("decrypt_fixed", |b| {
        b.iter(|| black_box(&prikey).decrypt(black_box(&ciphertext)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = paillier_benchmark
}
criterion_main!(benches);
