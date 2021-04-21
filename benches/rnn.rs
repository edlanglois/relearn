use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_rl::torch::configs::RnnConfig;
use rust_rl::torch::seq_modules::{SeqModRnn, SequenceModule};
use rust_rl::torch::ModuleBuilder;
use std::array::IntoIter;
use tch::{nn::VarStore, nn::GRU, Device, Kind, Tensor};

fn gru_seq_serial(c: &mut Criterion) {
    let batch_size = 1;
    let in_features = 3;
    let out_features = 4;
    let vs = VarStore::new(Device::Cpu);
    let gru: SeqModRnn<GRU> =
        RnnConfig::default().build_module(&vs.root(), in_features, out_features);

    // n episodes of length 1 in serial
    let mut ep_group = c.benchmark_group("gru_seq_serial_n_episodes");
    for total_steps in IntoIter::new([1usize, 10, 100, 1000]) {
        let input = Tensor::ones(
            &[batch_size, total_steps as i64, in_features as i64],
            (Kind::Float, Device::Cpu),
        );

        ep_group.throughput(Throughput::Elements(total_steps as u64));
        let seq_lengths = vec![1; total_steps];
        ep_group.bench_with_input(
            BenchmarkId::from_parameter(total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_serial(input, &seq_lengths)),
        );
    }
    ep_group.finish();

    // 1 episode of length n
    let mut step_group = c.benchmark_group("gru_seq_serial_n_steps");
    for total_steps in IntoIter::new([1usize, 10, 100, 1000]) {
        let input = Tensor::ones(
            &[batch_size, total_steps as i64, in_features as i64],
            (Kind::Float, Device::Cpu),
        );

        step_group.throughput(Throughput::Elements(total_steps as u64));
        let seq_lengths = [total_steps];
        step_group.bench_with_input(
            BenchmarkId::from_parameter(total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_serial(input, &seq_lengths)),
        );
    }
    step_group.finish();

    // n episodes of length 1 batched
    let mut ep_group = c.benchmark_group("gru_seq_serial_n_batches");
    for total_steps in IntoIter::new([1usize, 10, 100, 1000]) {
        let input = Tensor::ones(
            &[total_steps as i64, 1, in_features as i64],
            (Kind::Float, Device::Cpu),
        );

        ep_group.throughput(Throughput::Elements(total_steps as u64));
        let seq_lengths = [1];
        ep_group.bench_with_input(
            BenchmarkId::from_parameter(total_steps),
            &input,
            |b, input| b.iter_with_large_drop(|| gru.seq_serial(input, &seq_lengths)),
        );
    }
    ep_group.finish();
}

criterion_group!(benches, gru_seq_serial);
criterion_main!(benches);
