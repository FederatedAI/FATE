use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fate_tensor::math::BInt;
use fate_tensor::paillier;
use std::time::Duration;

fn paillier_benchmark(c: &mut Criterion) {
    let (sk, pk) = paillier::keygen(1024);
    let plaintext = paillier::PT(BInt::from_str_radix("1234567890987654321", 10));
    let ciphertext = pk.encrypt(&plaintext, true);
    let mut group = c.benchmark_group("paillier");

    group.bench_function("keygen-1024", |b| {
        b.iter(|| paillier::keygen(black_box(1024)))
    });
    group.bench_function("keygen-2048", |b| {
        b.iter(|| paillier::keygen(black_box(1024)))
    });
    group.bench_function("encrypt", |b| {
        b.iter(|| black_box(&pk).encrypt(black_box(&plaintext), true))
    });
    group.bench_function("decrypt", |b| {
        b.iter(|| black_box(&sk).decrypt(black_box(&ciphertext)))
    });
    group.bench_function("add ciphertext", |b| {
        b.iter(|| black_box(&ciphertext).add_ct(black_box(&ciphertext), black_box(&pk)))
    });
    group.bench_function("mul plaintext", |b| {
        b.iter(|| black_box(&ciphertext).mul_pt(black_box(&plaintext), black_box(&pk)))
    });
    group.bench_function("obfuscate", |b| {
        b.iter(|| black_box(&ciphertext).to_owned().obfuscate(black_box(&pk)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = paillier_benchmark
}
criterion_main!(benches);
