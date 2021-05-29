//! Rust benchmarks
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("empty");

    group.bench_function("empty", |b| b.iter(|| {}));
}

fn bool_to_int(c: &mut Criterion) {
    let mut group = c.benchmark_group("bool_to_int");

    group.bench_function("as", |b| b.iter(|| black_box(true) as usize));
    group.bench_function("into", |b| b.iter(|| -> usize { black_box(true).into() }));
}

fn int_to_bool(c: &mut Criterion) {
    let mut group = c.benchmark_group("int_to_bool");

    group.bench_function("ne/1", |b| b.iter(|| black_box(1) != 0));
    group.bench_function("ne/3", |b| b.iter(|| black_box(3) != 0));
    group.bench_function("try_into/1", |b| {
        b.iter(|| -> bool {
            match black_box(1_usize) {
                0 => false,
                1 => true,
                _ => panic!("invalid"),
            }
        })
    });
}

criterion_group!(benches, empty, bool_to_int, int_to_bool);
criterion_main!(benches);
