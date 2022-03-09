use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use relearn::torch::modules::{BuildModule, GruConfig, SequenceModule};
use tch::{Device, Kind, Tensor};

fn gru_rnn(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru_seq");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    let batch_size = 1;
    let in_features = 3;
    let out_features = 4;
    let gru = GruConfig::default().build_module(in_features, out_features, Device::Cpu);

    for total_steps in [1usize, 10, 100, 1000].into_iter() {
        group.throughput(Throughput::Elements(total_steps as u64));
        let input = Tensor::ones(
            &[batch_size, total_steps as i64, in_features as i64],
            (Kind::Float, Device::Cpu),
        );

        // n episodes of length 1 in serial
        let seq_lengths = vec![1; total_steps];
        group.bench_with_input(
            BenchmarkId::new("seq_serial_n_episodes", total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_serial(input, &seq_lengths)),
        );

        // 1 episode of length n
        let seq_lengths = [total_steps];
        group.bench_with_input(
            BenchmarkId::new("seq_serial_n_steps", total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_serial(input, &seq_lengths)),
        );

        // n episodes of length 1, batched
        let input = Tensor::ones(
            &[total_steps as i64, 1, in_features as i64],
            (Kind::Float, Device::Cpu),
        );
        let seq_lengths = [1];
        group.bench_with_input(
            BenchmarkId::new("seq_serial_n_batches", total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_serial(input, &seq_lengths)),
        );

        // n episodes of length 1, packed
        let input = Tensor::ones(
            &[total_steps as i64, in_features as i64],
            (Kind::Float, Device::Cpu),
        );
        let batch_sizes = Tensor::of_slice(&[total_steps as i64]);
        group.bench_with_input(
            BenchmarkId::new("seq_packed_n_episodes", total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_packed(input, &batch_sizes)),
        );
    }
}

criterion_group!(benches, gru_rnn);
criterion_main!(benches);
