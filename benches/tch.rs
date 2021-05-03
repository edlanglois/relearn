use criterion::{criterion_group, criterion_main, Criterion};
use tch::{Device, Kind, Tensor};

/// Torch multiply-then-sum
fn tch_mul_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("tch_mul_sum");
    let m1 = &Tensor::rand(&[10, 10000], (Kind::Float, Device::Cpu));
    let m2 = &m1.rand_like();

    group.bench_function("mul_sum", |b| b.iter(|| (m1 * m2).sum(m1.kind())));
    group.bench_function("dot", |b| {
        b.iter(|| (m1.flatten(0, -1)).dot(&m2.flatten(0, -1)))
    });
}

criterion_group!(benches, tch_mul_sum);
criterion_main!(benches);
