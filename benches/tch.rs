use criterion::{criterion_group, criterion_main, Criterion};
use tch::{Device, Kind, Tensor};

fn mul_sum(a: &Tensor, b: &Tensor) -> Tensor {
    let c = a * b;
    c.sum(c.kind())
}

fn flat_dot(a: &Tensor, b: &Tensor) -> Tensor {
    a.flatten(0, 1).dot(&b.flatten(0, -1))
}

/// Torch multiply-then-sum
fn tch_mul_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("tch_mul_sum");
    let m1 = &Tensor::rand(&[10, 10000], (Kind::Float, Device::Cpu));
    let m2 = &m1.rand_like();

    group.bench_function("mul_sum", |b| b.iter(|| mul_sum(m1, m2)));
    group.bench_function("dot", |b| b.iter(|| flat_dot(m1, m2)));

    assert_eq!(mul_sum(m1, m2), flat_dot(m1, m2));
}

criterion_group!(benches, tch_mul_sum);
criterion_main!(benches);
