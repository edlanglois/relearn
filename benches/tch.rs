use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use ndarray::{Array, IxDyn};
use std::array::IntoIter;
use std::convert::{TryFrom, TryInto};
use tch::{Device, IndexOp, Kind, Tensor};

/// Tensor creation
fn tensor_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_create");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for size in IntoIter::new([1, 100, 10_000, 1_000_000]) {
        group.throughput(Throughput::Elements(size));

        group.bench_with_input(BenchmarkId::new("empty", size), &size, |b, size| {
            b.iter_with_large_drop(|| Tensor::empty(&[*size as i64], (Kind::Int64, Device::Cpu)))
        });
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, size| {
            b.iter_with_large_drop(|| Tensor::zeros(&[*size as i64], (Kind::Int64, Device::Cpu)))
        });
    }

    let size = 1;
    group.throughput(Throughput::Elements(size));
    group.bench_with_input(
        BenchmarkId::new("empty_and_drop", size),
        &size,
        |b, size| b.iter(|| Tensor::empty(&[*size as i64], (Kind::Int64, Device::Cpu))),
    );
}

/// Tensor Copy and Fill
fn tensor_copy_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_copy_fill");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for size in IntoIter::new([1, 100, 10_000, 1_000_000]) {
        group.throughput(Throughput::Elements(size));

        let src = Tensor::rand(&[size as i64], (Kind::Float, Device::Cpu));
        let mut dst = src.empty_like();

        group.bench_function(BenchmarkId::new("copy", size), |b| b.iter(|| src.copy()));
        group.bench_function(BenchmarkId::new("copy_", size), |b| {
            b.iter(|| dst.copy_(&src))
        });
        group.bench_function(BenchmarkId::new("fill_", size), |b| {
            b.iter(|| dst.fill_(1.0))
        });
    }
}

/// Tensor detach clone
fn tensor_detach_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_detach_clone");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for size in IntoIter::new([1, 1_000_000]) {
        group.throughput(Throughput::Elements(size));

        let src = Tensor::rand(&[size as i64], (Kind::Float, Device::Cpu));

        group.bench_function(BenchmarkId::new("detach", size), |b| {
            b.iter_with_large_drop(|| src.detach())
        });
        group.bench_function(BenchmarkId::new("shallow_clone", size), |b| {
            b.iter_with_large_drop(|| src.shallow_clone())
        });
    }
}

/// Tensor indexing
fn tensor_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_indexing");

    let input = Tensor::rand(&[10, 10, 10], (Kind::Float, Device::Cpu));

    group.bench_function("index_1", |b| b.iter_with_large_drop(|| input.i(0)));
    group.bench_function("index_2", |b| b.iter_with_large_drop(|| input.i((0, 0))));
    group.bench_function("index_2_drop", |b| b.iter(|| input.i((0, 0))));
    group.bench_function("index_3", |b| b.iter_with_large_drop(|| input.i((0, 0, 0))));
}

fn tensor_ndarray_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_ndarray_convert");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for size in IntoIter::new([1usize, 100, 10_000, 1_000_000]) {
        group.throughput(Throughput::Elements(size as u64));

        let array: Array<f32, _> = Array::ones(size);
        let tensor = Tensor::ones(&[size as i64], (Kind::Float, Device::Cpu));

        group.bench_function(BenchmarkId::new("from_array", size), |b| {
            b.iter(|| Tensor::try_from(&array).unwrap())
        });
        group.bench_function(BenchmarkId::new("to_array", size), |b| {
            b.iter(|| -> Array<f32, IxDyn> { (&tensor).try_into().unwrap() })
        });
    }
}

fn tensor_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_scatter");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    let width = 10_000;
    let mut target = Tensor::zeros(&[100, width], (Kind::Float, Device::Cpu));
    let source = target.empty_like().random_();

    for size in IntoIter::new([1usize, 100, 10_000, 1_000_000]) {
        group.throughput(Throughput::Elements(size as u64));
        let indices = if size >= 100 {
            Tensor::empty(&[100, (size / 100) as i64], (Kind::Int64, Device::Cpu)).random_1(width)
        } else {
            Tensor::empty(&[1, size as i64], (Kind::Int64, Device::Cpu)).random_1(width)
        };

        group.bench_function(BenchmarkId::new("scatter", size), |b| {
            b.iter(|| target.scatter(-1, &indices, &source))
        });
        group.bench_function(BenchmarkId::new("scatter1", size), |b| {
            b.iter(|| target.scatter1(-1, &indices, 1.0))
        });
        group.bench_function(BenchmarkId::new("scatter_", size), |b| {
            b.iter(|| target.scatter_(-1, &indices, &source))
        });
        group.bench_function(BenchmarkId::new("scatter1_", size), |b| {
            b.iter(|| target.scatter_1(-1, &indices, 1.0))
        });
    }
}

fn mul_sum(a: &Tensor, b: &Tensor) -> Tensor {
    let c = a * b;
    c.sum(c.kind())
}

fn flat_dot(a: &Tensor, b: &Tensor) -> Tensor {
    a.flatten(0, 1).dot(&b.flatten(0, -1))
}

/// Torch multiply-then-sum
fn tensor_mul_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_mul_sum");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for size in IntoIter::new([1, 100, 10_000, 1_000_000]) {
        group.throughput(Throughput::Elements(size as u64));
        let shape = if size >= 10 {
            [10, size / 10]
        } else {
            [1, size]
        };

        let m1 = &Tensor::rand(&shape, (Kind::Float, Device::Cpu));
        let m2 = &m1.rand_like();

        group.bench_function(BenchmarkId::new("mul_sum", size), |b| {
            b.iter(|| mul_sum(m1, m2))
        });
        group.bench_function(BenchmarkId::new("dot", size), |b| {
            b.iter(|| flat_dot(m1, m2))
        });

        assert!(bool::from(
            mul_sum(m1, m2)
                .isclose(&flat_dot(m1, m2), 1e-6, 1e-6, false)
                .all()
        ));
    }
}

criterion_group!(
    benches,
    tensor_create,
    tensor_copy_fill,
    tensor_detach_clone,
    tensor_indexing,
    tensor_ndarray_convert,
    tensor_scatter,
    tensor_mul_sum,
);
criterion_main!(benches);
