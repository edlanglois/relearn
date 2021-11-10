//! Spaces benchmarks
use criterion::{
    criterion_group, criterion_main, measurement::Measurement, AxisScale, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use relearn::spaces::{
    BatchFeatureSpace, BooleanSpace, IndexSpace, IntervalSpace, NonEmptyFeatures, OptionSpace,
    PowerSpace, ProductSpace, SampleSpace, SingletonSpace,
};
use tch::Tensor;

fn bench_space_batch_tensor_features<S, M>(
    group: &mut BenchmarkGroup<M>,
    name: &str,
    space: S,
    batch_sizes: &[u64],
) where
    S: SampleSpace + BatchFeatureSpace<Tensor>,
    M: Measurement,
{
    let mut rng = StdRng::seed_from_u64(0);

    for &batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size));

        let data: Vec<S::Element> = (&mut rng)
            .sample_iter(&space)
            .take(batch_size as usize)
            .collect();
        group.bench_with_input(BenchmarkId::new(name, batch_size), &data, |b, data| {
            b.iter_with_large_drop(|| space.batch_features(data))
        });
    }
}

fn bench_batch_tensor_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_tensor_features");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    let batch_sizes = [1, 100, 10_000, 1_000_000];

    bench_space_batch_tensor_features(&mut group, "boolean", BooleanSpace, &batch_sizes);
    bench_space_batch_tensor_features(&mut group, "index_10", IndexSpace::new(10), &batch_sizes);
    bench_space_batch_tensor_features(
        &mut group,
        "interval_inf",
        IntervalSpace::<f64>::default(),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "interval_01",
        IntervalSpace::new(0.0, 1.0),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "nonempty_boolean",
        NonEmptyFeatures::new(BooleanSpace),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "nonempty_singleton",
        NonEmptyFeatures::new(SingletonSpace),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "option_index",
        OptionSpace::new(IndexSpace::new(10)),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "power_d0_boolean",
        PowerSpace::<_, 0>::new(BooleanSpace),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "power_d3_boolean",
        PowerSpace::<_, 3>::new(BooleanSpace),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(
        &mut group,
        "product_boolean_index_singleton",
        ProductSpace::new((BooleanSpace, IndexSpace::new(10), SingletonSpace)),
        &batch_sizes,
    );
    bench_space_batch_tensor_features(&mut group, "singleton", SingletonSpace, &batch_sizes);
}

criterion_group!(benches, bench_batch_tensor_features);
criterion_main!(benches);
