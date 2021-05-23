use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use ndarray::{s, Array, Array2, ArrayViewMut, Ix1};
use std::array::IntoIter;
use std::convert::TryInto;
use tch::{Device, IndexOp, Kind, Tensor};

fn split_option_array(elements: &[Option<usize>]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut none_indices = Vec::new();
    let mut some_indices = Vec::new();
    let mut some_values = Vec::new();
    for (i, element) in elements.iter().enumerate() {
        if let Some(x) = element {
            some_indices.push(i as i64);
            some_values.push(*x as i64);
        } else {
            none_indices.push(i as i64);
        }
    }
    (none_indices, some_indices, some_values)
}

fn one_hot(labels: &Tensor, num_classes: usize, kind: Kind) -> Tensor {
    let mut shape = labels.size();
    shape.push(num_classes as i64);
    Tensor::zeros(&shape, (kind, labels.device())).scatter_1(-1, &labels.unsqueeze(-1), 1)
}

pub fn one_hot_out(labels: &Tensor, out: &mut Tensor) {
    let _ = out.zero_();
    let _ = out.scatter_1(-1, &labels.unsqueeze(-1), 1);
}

fn option_index_features_by_copy(elements: &[Option<usize>], num_classes: usize) -> Tensor {
    let (none_indices, some_indices, some_values) = split_option_array(&elements);
    let mut out = Tensor::empty(
        &[elements.len() as i64, (num_classes + 1) as i64],
        (Kind::Float, Device::Cpu),
    );
    let _ = out.zero_();
    let _ = out
        .i((.., 0))
        .index_fill_(-1, &Tensor::of_slice(&none_indices), 1.0);
    let some_features = one_hot(&Tensor::of_slice(&some_values), num_classes, Kind::Float);
    let _ = out
        .i((.., 1..))
        .index_copy_(-2, &Tensor::of_slice(&some_indices), &some_features);
    out
}

fn option_index_features_by_stack_cat_rows(
    elements: &[Option<usize>],
    num_classes: usize,
) -> Tensor {
    let mut rows = Vec::new();
    let kind = Kind::Float;
    let device = Device::Cpu;
    for element in elements {
        if let Some(x) = element {
            let is_none = Tensor::zeros(&[1], (kind, device));
            let inner_features = one_hot(
                &Tensor::scalar_tensor(*x as i64, (Kind::Int64, device)),
                num_classes,
                kind,
            );
            rows.push(Tensor::cat(&[is_none, inner_features], -1));
        } else {
            let row = Tensor::zeros(&[(num_classes + 1) as i64], (kind, device));
            let _ = row.i(0).fill_(1.0);
            rows.push(row);
        };
    }
    Tensor::vstack(&rows)
}

fn option_index_features_by_row_empty_unbind_split(
    elements: &[Option<usize>],
    num_classes: usize,
) -> Tensor {
    let out = Tensor::empty(
        &[elements.len() as i64, (num_classes + 1) as i64],
        (Kind::Float, Device::Cpu),
    );
    for (row, element) in out.unbind(0).iter().zip(elements) {
        let [mut is_none, mut inner_features]: [Tensor; 2] = row
            .split_with_sizes(&[1, num_classes as i64], -1)
            .try_into()
            .unwrap();
        if let Some(x) = element {
            let _ = is_none.zero_();
            one_hot_out(
                &Tensor::scalar_tensor(*x as i64, (Kind::Int64, Device::Cpu)),
                &mut inner_features,
            );
        } else {
            let _ = is_none.fill_(1.0);
            let _ = inner_features.zero_();
        }
    }
    out
}

fn option_index_features_by_row_zeros_unbind_split(
    elements: &[Option<usize>],
    num_classes: usize,
) -> Tensor {
    let out = Tensor::zeros(
        &[elements.len() as i64, (num_classes + 1) as i64],
        (Kind::Float, Device::Cpu),
    );
    for (row, element) in out.unbind(0).iter().zip(elements) {
        let [mut is_none, mut inner_features]: [Tensor; 2] = row
            .split_with_sizes(&[1, num_classes as i64], -1)
            .try_into()
            .unwrap();
        if let Some(x) = element {
            one_hot_out(
                &Tensor::scalar_tensor(*x as i64, (Kind::Int64, Device::Cpu)),
                &mut inner_features,
            );
        } else {
            let _ = is_none.fill_(1.0);
        }
    }
    out
}

fn option_index_features_by_row_zeros_split_unbind(
    elements: &[Option<usize>],
    num_classes: usize,
) -> Tensor {
    let out = Tensor::zeros(
        &[elements.len() as i64, (num_classes + 1) as i64],
        (Kind::Float, Device::Cpu),
    );

    let mut none_indices = Vec::new();
    for (i, (mut inner_row, element)) in out
        .i((.., 1..))
        .unbind(0)
        .into_iter()
        .zip(elements)
        .enumerate()
    {
        if let Some(x) = element {
            one_hot_out(
                &Tensor::scalar_tensor(*x as i64, (Kind::Int64, Device::Cpu)),
                &mut inner_row,
            );
        } else {
            none_indices.push(i as i64);
        }
    }
    let _ = out
        .i((.., 0))
        .index_fill_(0, &Tensor::of_slice(&none_indices), 1.0);
    out
}

fn option_index_features_by_masked_out(elements: &[Option<usize>], num_classes: usize) -> Tensor {
    let mut out = Tensor::zeros(
        &[elements.len() as i64, (num_classes + 1) as i64],
        (Kind::Float, Device::Cpu),
    );
    let is_none: Vec<_> = elements.iter().map(Option::is_none).collect();
    let is_none_tensor = Tensor::of_slice(&is_none);

    let _ = out.i((.., 0)).masked_fill_(&is_none_tensor, 1.0);

    // As though done within IndexSpace::batch_features_masked_out(some_values, &out, ...)
    let shifted_indices: Vec<_> = elements
        .iter()
        .map(|x| (x.unwrap_or(0) + 1) as i64)
        .collect();
    let shifted_index_tensor = Tensor::of_slice(&shifted_indices).unsqueeze(-1);
    let _ = out.scatter_(
        -1,
        &shifted_index_tensor,
        &is_none_tensor
            .logical_not()
            .to_kind(Kind::Float)
            .unsqueeze(-1),
    );
    out
}

fn array_one_hot(array: &mut ArrayViewMut<f32, Ix1>, index: usize) {
    array[index] = 1.0;
}

fn option_index_ndarray_features(elements: &[Option<usize>], num_classes: usize) -> Array2<f32> {
    let mut out = Array::zeros((elements.len(), num_classes + 1));
    for (mut row, element) in out.outer_iter_mut().zip(elements) {
        if let Some(x) = element {
            array_one_hot(&mut row.slice_mut(s![1..]), *x);
        } else {
            row[0] = 1.0;
        }
    }
    out
}

fn option_index_features_by_ndarray(elements: &[Option<usize>], num_classes: usize) -> Tensor {
    option_index_ndarray_features(elements, num_classes)
        .try_into()
        .unwrap()
}

/// Test different possible implementations for OptionSpace::batch_features
fn option_index_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("option_index_features");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    let num_classes = 6;

    for size in IntoIter::new([1, 100, 10_000, 1_000_000]) {
        group.throughput(Throughput::Elements(size as u64));
        let elements: Vec<_> = (0..size)
            .map(|i| {
                if (i % 5 == 0) || (i % 7 == 0) {
                    None
                } else {
                    Some(i % num_classes)
                }
            })
            .collect();

        let r1 = option_index_features_by_copy(&elements, num_classes);
        group.bench_function(BenchmarkId::new("by_copy", size), |b| {
            b.iter(|| option_index_features_by_copy(&elements, num_classes))
        });

        let r2 = option_index_features_by_stack_cat_rows(&elements, num_classes);
        assert_eq!(r1, r2);
        group.bench_function(BenchmarkId::new("by_stack_cat_rows", size), |b| {
            b.iter(|| option_index_features_by_stack_cat_rows(&elements, num_classes))
        });

        let r3 = option_index_features_by_row_empty_unbind_split(&elements, num_classes);
        assert_eq!(r1, r3);
        group.bench_function(BenchmarkId::new("by_row_empty_unbind_split", size), |b| {
            b.iter(|| option_index_features_by_row_empty_unbind_split(&elements, num_classes))
        });

        let r4 = option_index_features_by_row_zeros_unbind_split(&elements, num_classes);
        assert_eq!(r1, r4);
        group.bench_function(BenchmarkId::new("by_row_zeros_unbind_split", size), |b| {
            b.iter(|| option_index_features_by_row_zeros_unbind_split(&elements, num_classes))
        });

        let r5 = option_index_features_by_row_zeros_split_unbind(&elements, num_classes);
        assert_eq!(r1, r5);
        group.bench_function(BenchmarkId::new("by_row_zeros_split_unbind", size), |b| {
            b.iter(|| option_index_features_by_row_zeros_split_unbind(&elements, num_classes))
        });

        let r6 = option_index_features_by_masked_out(&elements, num_classes);
        assert_eq!(r1, r6);
        group.bench_function(BenchmarkId::new("by_masked_out", size), |b| {
            b.iter(|| option_index_features_by_masked_out(&elements, num_classes))
        });

        group.bench_function(BenchmarkId::new("ndarray", size), |b| {
            b.iter(|| option_index_ndarray_features(&elements, num_classes))
        });

        let r7 = option_index_features_by_ndarray(&elements, num_classes);
        assert_eq!(r1, r7);
        group.bench_function(BenchmarkId::new("by_ndarray", size), |b| {
            b.iter(|| option_index_features_by_ndarray(&elements, num_classes))
        });
    }
}

criterion_group!(benches, option_index_features);
criterion_main!(benches);
