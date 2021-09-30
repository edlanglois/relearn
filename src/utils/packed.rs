//! Utilities for dealing with packed sequences.
//!
//! A packed set of sequences is an array representing multiple heterogeneous-length sequences
//! where the data is ordered first by offset within a sequence then by sequence index.
//!
//! Sequences must always be packed in order from longest to shortest.
//!
//! For example, the sequences `[0, 1, 2, 3]`, `[10, 11]`, `[100, 101]` are packed as
//! `[0, 10, 100, 1, 11, 101, 2, 3]`.
use std::iter;
use std::ops::{Index, Range};
use tch::{IndexOp, Scalar, Tensor};

/// Iterator that packs together multiple sequences.
///
/// # Example
/// ```
/// use rust_rl::utils::packed::PackedIter;
///
/// let sequences = [vec![0, 1, 2, 3], vec![10, 11], vec![100, 101]];
/// let packed: Vec<_> = PackedIter::from_sorted(&sequences).map(|&x| x).collect();
/// assert_eq!(packed, vec![0, 10, 100, 1, 11, 101, 2, 3]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackedIter<T> {
    /// Sequences from which elements are obtained.
    sequences: Vec<T>,
    /// Current sequence index
    seq_idx: usize,
    /// Number of sequences that might still have elements.
    batch_size: usize,
}

impl<T> PackedIter<T>
where
    T: ExactSizeIterator,
{
    /// Create a new [`PackedIter`] instance.
    ///
    /// # Args
    /// * `sequences` - A iterator of sequence iterators.
    ///     The inner sequence iterators must have fixed lengths.
    ///
    /// # Note
    /// Sequences will be reordered (stably) so that
    /// they are in order of monotonically decreasing length.
    pub fn new<U>(sequences: U) -> Self
    where
        U: IntoIterator,
        U::Item: IntoIterator<IntoIter = T>,
    {
        let mut sequences: Vec<_> = sequences.into_iter().map(IntoIterator::into_iter).collect();
        // Sort in descending order of length.
        sequences.sort_by(|a, b| a.len().cmp(&b.len()).reverse());
        let batch_size = sequences.len();
        Self {
            sequences,
            seq_idx: 0,
            batch_size,
        }
    }

    /// Create a new [`PackedIter`] from sequences sorted in descending order of length.
    ///
    /// # Args
    /// * `sequences` - A iterator of sequence iterators.
    ///     The inner sequence iterators must have fixed lengths
    ///     and be sorted in order of monotonically decreasing length.
    pub fn from_sorted<U>(sequences: U) -> Self
    where
        U: IntoIterator,
        U::Item: IntoIterator<IntoIter = T>,
    {
        let sequences: Vec<_> = sequences.into_iter().map(IntoIterator::into_iter).collect();
        let batch_size = sequences.len();
        Self {
            sequences,
            seq_idx: 0,
            batch_size,
        }
    }
}

impl<T> Iterator for PackedIter<T>
where
    T: ExactSizeIterator,
{
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = self.sequences[self.seq_idx].next();
        if result.is_none() {
            if self.batch_size == 0 {
                return None;
            }
            self.batch_size = self.seq_idx;
            self.seq_idx = 0;
            result = self.sequences[self.seq_idx].next();
        }
        self.seq_idx += 1;
        if self.seq_idx >= self.batch_size {
            self.seq_idx = 0;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.sequences.iter().map(|s| s.len()).sum();
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for PackedIter<T> where T: ExactSizeIterator {}

/// Iterator of indices that pack the specified contiguous sequences.
///
/// # Example
/// ```
/// use rust_rl::utils::packed::PackingIndices;
///
/// let data = [0, 1, 2, 3, 10, 11, 100, 101];
/// let ranges = [0..4, 4..6, 6..8];
/// let packed: Vec<_> = PackingIndices::from_sorted(&ranges).map(|i| data[i]).collect();
/// assert_eq!(packed, vec![0, 10, 100, 1, 11, 101, 2, 3]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackingIndices<'a> {
    /// Sequence index ranges, sorted in monotonically decreasing order of sequence length.
    sequence_ranges: &'a [Range<usize>],

    /// Current offset within the sequences.
    offset: usize,
    /// Current sequence index within `sequences`.
    seq_idx: usize,
    /// Number of sequences with length longer than the current offset.
    batch_size: usize,
}

impl<'a> PackingIndices<'a> {
    /// Create a new [`PackingIndices`] instance from ranges sorted in decreasing order of length.
    ///
    /// # Args
    /// * `sequence_ranges` - Index range for each sequence,
    ///     sorted in monotonically decreasing order of sequence length.
    pub fn from_sorted(sequence_ranges: &'a [Range<usize>]) -> Self {
        let mut batch_size = sequence_ranges.len();
        let offset = 0;
        update_batch_size(sequence_ranges, offset, &mut batch_size);
        Self {
            sequence_ranges,
            offset,
            seq_idx: 0,
            batch_size,
        }
    }
}

impl<'a> Iterator for PackingIndices<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_size == 0 {
            return None;
        }
        let index = self.sequence_ranges[self.seq_idx].start + self.offset;
        self.seq_idx += 1;
        if self.seq_idx >= self.batch_size {
            self.seq_idx = 0;
            self.offset += 1;
            update_batch_size(self.sequence_ranges, self.offset, &mut self.batch_size);
        }
        Some(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self
            .sequence_ranges
            .iter()
            .take(self.batch_size)
            .map(|range| range.len() - self.offset)
            .sum::<usize>()
            - self.seq_idx;
        (size, Some(size))
    }
}

impl<'a> ExactSizeIterator for PackingIndices<'a> {}

/// The batch size of each offset when sequences of the given lengths are packed.
///
/// # Example
/// ```
/// use rust_rl::utils::packed::PackedBatchSizes;
///
/// let sequence_lengths = [4, 2, 2];
/// let batch_sizes: Vec<_> = PackedBatchSizes::from_sorted_lengths(&sequence_lengths).collect();
/// assert_eq!(batch_sizes, vec![3, 3, 1, 1]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedBatchSizes<'a, T> {
    /// Sequences lengths sorted in monotonically decreasing order.
    sequence_lengths: &'a [T],
    /// Current offset within the sequences.
    offset: usize,
    /// Number of sequences with length > offset
    batch_size: usize,
}

impl<'a> PackedBatchSizes<'a, usize> {
    /// Create a [`PackedBatchSizes`] instance from monotonically decreasing sequence lengths.
    pub fn from_sorted_lengths(sequence_lengths: &'a [usize]) -> Self {
        let offset = 0;
        let mut batch_size = sequence_lengths.len();
        update_batch_size(sequence_lengths, offset, &mut batch_size);
        Self {
            sequence_lengths,
            offset,
            batch_size,
        }
    }
}

impl<'a> PackedBatchSizes<'a, Range<usize>> {
    /// Create a [`PackedBatchSizes`] instance from ranges with monotonically decreasing lengths.
    pub fn from_sorted_ranges(sequence_ranges: &'a [Range<usize>]) -> Self {
        let offset = 0;
        let mut batch_size = sequence_ranges.len();
        update_batch_size(sequence_ranges, offset, &mut batch_size);
        Self {
            sequence_lengths: sequence_ranges,
            offset,
            batch_size,
        }
    }
}

impl<'a, T> Iterator for PackedBatchSizes<'a, T>
where
    T: Length,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_size == 0 {
            None
        } else {
            let current_batch_size = self.batch_size;
            self.offset += 1;
            update_batch_size(self.sequence_lengths, self.offset, &mut self.batch_size);
            Some(current_batch_size)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let length = match self.sequence_lengths.first() {
            Some(v) => v.length() - self.offset,
            None => 0,
        };
        (length, Some(length))
    }
}

impl<'a, T> ExactSizeIterator for PackedBatchSizes<'a, T> where T: Length {}

/// Decrement batch size until it is the batch size of the given offset.
fn update_batch_size<T>(sorted_sequence_ranges: &T, offset: usize, batch_size: &mut usize)
where
    T: Index<usize> + ?Sized,
    <T as Index<usize>>::Output: Length,
{
    while *batch_size > 0 && sorted_sequence_ranges[*batch_size - 1].length() <= offset {
        *batch_size -= 1;
    }
}

/// Has a length
pub trait Length {
    fn length(&self) -> usize;
}

impl Length for Range<usize> {
    fn length(&self) -> usize {
        self.len()
    }
}

/// Directly stores a length
impl Length for usize {
    fn length(&self) -> usize {
        *self
    }
}

/// View each batched offset of a packed tensor.
///
/// # Args
/// `packed`: A tensor of sequences packed along the first dimension.
/// `batch_sizes`: Batch sizes along the first dimension of `packed`.
///                The tch interface requires that this be i64.
///
/// # Returns
/// A vector contained tensor view of each batched time step of the sequences in `packed`.
/// Has length equal to `batch_sizes`.
/// Each tensor in the output has the same shape as `packed` except for the first dimension
/// which has length `batch_sizes[i]`.
pub fn packed_tensor_batched_steps(packed: &Tensor, batch_sizes: &[i64]) -> Vec<Tensor> {
    packed.split_with_sizes(batch_sizes, 0)
}

/// A view of a packed tensor starting from the given offset within each sequence.
///
/// Does not make any copies of the underlying tensors.
///
/// # Args
/// * `packed`: A tensor of sequences packed along the first dimension.
/// * `batch_sizes`: Batch sizes along the first dimension of `packed`.
/// * `offset`: Sequence offset from which to start the view.
///
/// # Returns
/// * `packed`: Packed values from `packed` excluding the first `offset` steps of each
///     sequence. Has the same shape as `packed` except for the first dimension.
/// * `batch_sizes`: Batch sizes for the output packed tensor.

pub fn packed_tensor_from_offset<'a>(
    packed: &Tensor,
    batch_sizes: &'a [i64],
    offset: usize,
) -> (Tensor, &'a [i64]) {
    let packed_offset = batch_sizes[..offset].iter().sum::<i64>() as i64;
    (packed.i(packed_offset..), &batch_sizes[offset..])
}

/// Collect batches into groups where only the last batch in a group changes size.
///
/// This is designed for resizes that affect only the end of a batch.
///
/// If `old_batch_sizes` and `new_batch_sizes` have different lengths then any excess are grouped
/// together and it is assumed that the corresponding new/old batch size is 0.
///
/// # Args
/// * `old_batch_sizes` - An iterator over the old/original batch sizes.
/// * `new_batch_sizes` - An iterator over the new batch sizes.
///
/// # Returns
/// * `old_group_sizes` - A vector of old group sizes.
///                       Each group consists of one or more batches to be resized at the end.
/// * `new_group_sizes` - A vector with the new size for each group in `old_group_sizes`.
///                       Has the same length as `old_group_sizes`.
///
fn group_batches_for_resize<'a, 'b, T, U>(
    old_batch_sizes: T,
    new_batch_sizes: U,
) -> (Vec<i64>, Vec<i64>)
where
    T: IntoIterator<Item = &'a i64>,
    U: IntoIterator<Item = &'b i64>,
{
    let mut old_group_sizes: Vec<i64> = Vec::new(); // i64 required by tch interface
    let mut new_group_sizes: Vec<i64> = Vec::new();
    let mut next_old_group_size = 0;
    let mut next_new_group_size = 0;

    let mut old_batch_size_iter = old_batch_sizes.into_iter().fuse();
    let mut new_batch_size_iter = new_batch_sizes.into_iter().fuse();
    loop {
        let (old, new, tail) = match (old_batch_size_iter.next(), new_batch_size_iter.next()) {
            (Some(old), Some(new)) => (*old, *new, false),
            (Some(old), None) => (*old, 0, true),
            (None, Some(new)) => (0, *new, true),
            (None, None) => break,
        };
        next_old_group_size += old;
        next_new_group_size += new;
        // Push the merged groups if the sizes differ.
        // Can merge consecutive tail batches because the whole tail will be added/removed
        // so it is still a suffix operation on the group, just one that affects multiple batches.
        if !tail && old != new {
            old_group_sizes.push(next_old_group_size);
            next_old_group_size = 0;
            new_group_sizes.push(next_new_group_size);
            next_new_group_size = 0;
        }
    }
    if next_old_group_size != 0 || next_new_group_size != 0 {
        old_group_sizes.push(next_old_group_size as i64);
        new_group_sizes.push(next_new_group_size as i64);
    }
    (old_group_sizes, new_group_sizes)
}

/// Copy a packed tensor without the last n elements of each sequence.
///
/// # Args
/// * `packed`: A tensor of sequences packed along the first dimension.
/// * `batch_sizes`: Batch sizes along the first dimension of `packed`.
///
/// # Returns
/// * `packed`: Packed values from `packed` excluding the last `n` steps of each
///     sequence. Has the same shape as `packed` except for the first dimension.
/// * `batch_sizes`: Batch sizes for the output packed tensor.
///
pub fn packed_tensor_trim_end<'a>(
    packed: &Tensor,
    batch_sizes: &'a [i64],
    n: usize,
) -> (Tensor, &'a [i64]) {
    // Batch sizes are the same as if we had dropped from the start of each sequence
    let new_batch_sizes = &batch_sizes[n..];

    let (old_group_sizes, new_group_sizes) = group_batches_for_resize(batch_sizes, new_batch_sizes);
    let groups = packed.split_with_sizes(&old_group_sizes, 0);

    let new_groups: Vec<_> = groups
        .iter()
        .zip(new_group_sizes)
        .map(|(group, new_size)| group.i(..new_size))
        .collect();
    let new_packed = Tensor::cat(&new_groups, 0);

    (new_packed, new_batch_sizes)
}

/// Copy a packed tensor with a value append to every sequence and the first value dropped.
///
/// The sequence lengths and batch sizes remain unchanged.
///
/// # Args
/// * `packed`: A tensor of sequences packed along the first dimension.
/// * `batch_sizes`: Batch sizes along the first dimension of `packed`.
///
/// # Returns
/// Tensor of output packed sequences. Has the same shape and batch sizes as `packed`.
///
pub fn packed_tensor_push_shift<S>(packed: &Tensor, batch_sizes: &[i64], value: S) -> Tensor
where
    S: Into<Scalar> + Copy,
{
    let (old_group_sizes, new_group_sizes) =
        group_batches_for_resize(batch_sizes, iter::once(&0).chain(batch_sizes.iter()));

    let input_groups = packed.split_with_sizes(&old_group_sizes, 0);

    let packed_output = packed.empty_like();
    let output_groups = packed_output.split_with_sizes(&new_group_sizes[1..], 0);

    // Drop the first input group
    // Then each input group is copied to an output group padded with value.
    // Finally, the remaining out_group is filled with value

    let mut out_group_iter = output_groups.into_iter().fuse();
    for ((in_group, &in_group_size), out_group) in input_groups[1..]
        .iter()
        .zip(&old_group_sizes[1..])
        .zip(&mut out_group_iter)
    {
        out_group.i(..in_group_size).copy_(in_group);
        let _ = out_group.i(in_group_size..).fill_(value);
    }
    for mut out_group in out_group_iter {
        let _ = out_group.fill_(value);
    }

    packed_output
}

/// Evaluate a discounted cumulative sum from sequence end to start on a packed tensor.
///
/// For each element `x[i]` in the sequence `x[0] ... x[N]`,
/// returns `y[i] = sum_{j in i..N} discount_factor ** (j - i) * x[j]`
///
/// # Args
/// * `packed`: A tensor of sequences packed along the first dimension.
/// * `batch_sizes`: Batch sizes along the first dimension of `packed`.
/// * `discount_factor`: Discount factor to apply in the cumulative sum.
///
/// # Returns
/// A tensor with the same shape as `packed` containing the discounted cumulative sums.
// TODO: Consider a recursive divide-and-conquer version for better parallelism
pub fn packed_tensor_discounted_cumsum_from_end(
    packed: &Tensor,
    batch_sizes: &[i64],
    discount_factor: f64,
) -> Tensor {
    if batch_sizes.is_empty() {
        assert_eq!(packed.numel(), 0);
        return packed.zeros_like();
    }

    let packed_output = packed.empty_like();

    let in_steps = packed_tensor_batched_steps(packed, batch_sizes);
    let mut out_steps = packed_tensor_batched_steps(&packed_output, batch_sizes);

    let max_batch_size = batch_sizes[0];
    let accumulator = Tensor::zeros(&[max_batch_size], (packed.kind(), packed.device()));
    for ((in_step, out_step), &batch_size) in in_steps
        .iter()
        .rev()
        .zip(out_steps.iter_mut().rev())
        .zip(batch_sizes.iter().rev())
    {
        let mut batch_accumulator = accumulator.i(..batch_size);
        batch_accumulator *= discount_factor;
        batch_accumulator += in_step;
        out_step.copy_(&batch_accumulator);
    }

    packed_output
}

#[cfg(test)]
mod packed_iter {
    use super::*;
    use std::array::IntoIter;

    #[test]
    fn iter_from_sorted() {
        let sequences = [vec![0, 1, 2, 3], vec![10, 11], vec![100, 101]];
        let packed: Vec<_> = PackedIter::from_sorted(IntoIter::new(sequences)).collect();
        let expected = vec![0, 10, 100, 1, 11, 101, 2, 3];
        assert_eq!(packed, expected);
    }

    #[test]
    fn iter_from_unsorted() {
        let sequences = [vec![10, 11], vec![0, 1, 2, 3], vec![100, 101]];
        let packed: Vec<_> = PackedIter::new(IntoIter::new(sequences)).collect();
        // First sorts the sequences then packs them.
        let expected = vec![0, 10, 100, 1, 11, 101, 2, 3];
        assert_eq!(packed, expected);
    }

    #[test]
    fn size_hint() {
        let sequences = [vec![10, 11], vec![0, 1, 2, 3], vec![100, 101]];
        let packed_iter = PackedIter::new(&sequences);
        assert_eq!(packed_iter.size_hint(), (8, Some(8)));
    }

    #[test]
    fn size_hint_after_next() {
        let sequences = [vec![10, 11], vec![0, 1, 2, 3], vec![100, 101]];
        let mut packed_iter = PackedIter::new(&sequences);
        let _ = packed_iter.next();
        assert_eq!(packed_iter.size_hint(), (7, Some(7)));
        let _ = packed_iter.next();
        assert_eq!(packed_iter.size_hint(), (6, Some(6)));
    }
}

#[cfg(test)]
mod packing_indices {
    use super::*;

    #[test]
    fn iter() {
        let data = [0, 1, 2, 3, 10, 11, 100, 101];
        let ranges = [0..4, 4..6, 6..8];
        let packed: Vec<_> = PackingIndices::from_sorted(&ranges)
            .map(|i| data[i])
            .collect();
        let expected = vec![0, 10, 100, 1, 11, 101, 2, 3];
        assert_eq!(packed, expected);
    }

    #[test]
    fn size_hint() {
        let ranges = [0..4, 4..6, 6..8];
        let packing_indices = PackingIndices::from_sorted(&ranges);
        assert_eq!(packing_indices.size_hint(), (8, Some(8)));
    }

    #[test]
    fn size_hint_after_next() {
        let ranges = [0..4, 4..6, 6..8];
        let mut packing_indices = PackingIndices::from_sorted(&ranges);
        let _ = packing_indices.next();
        assert_eq!(packing_indices.size_hint(), (7, Some(7)));
        let _ = packing_indices.next();
        assert_eq!(packing_indices.size_hint(), (6, Some(6)));
    }
}

#[cfg(test)]
mod packed_batch_sizes {
    use super::*;

    #[test]
    fn iter_from_lengths() {
        let sequence_lengths = [4, 2, 2];
        let packed: Vec<_> = PackedBatchSizes::from_sorted_lengths(&sequence_lengths).collect();
        let expected = vec![3, 3, 1, 1];
        assert_eq!(packed, expected);
    }

    #[test]
    fn iter_from_ranges() {
        let ranges = [0..4, 4..6, 6..8];
        let packed: Vec<_> = PackedBatchSizes::from_sorted_ranges(&ranges).collect();
        let expected = vec![3, 3, 1, 1];
        assert_eq!(packed, expected);
    }

    #[test]
    fn size_hint() {
        let sequence_lengths = [4, 2, 2];
        let packed_batch_sizes = PackedBatchSizes::from_sorted_lengths(&sequence_lengths);
        assert_eq!(packed_batch_sizes.size_hint(), (4, Some(4)));
    }

    #[test]
    fn size_hint_after_next() {
        let sequence_lengths = [4, 2, 2];
        let mut packed_batch_sizes = PackedBatchSizes::from_sorted_lengths(&sequence_lengths);
        let _ = packed_batch_sizes.next();
        assert_eq!(packed_batch_sizes.size_hint(), (3, Some(3)));
        let _ = packed_batch_sizes.next();
        assert_eq!(packed_batch_sizes.size_hint(), (2, Some(2)));
    }
}

#[cfg(test)]
mod packed_tensor {
    use super::*;

    #[test]
    fn from_offset_1() {
        // Sequences: [0, 1, 2, 3], [10, 11], [100, 101]
        let packed = Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]);
        let batch_sizes = [3, 3, 1, 1];
        let (packed_out, batch_sizes_out) = packed_tensor_from_offset(&packed, &batch_sizes, 1);
        // Sequences: [1, 2, 3], [11], [101]
        assert_eq!(packed_out, Tensor::of_slice(&[1, 11, 101, 2, 3]));
        assert_eq!(batch_sizes_out, &[3, 1, 1]);
    }

    #[test]
    fn from_offset_is_view() {
        // Sequences: [0, 1, 2, 3], [10, 11], [100, 101]
        let packed = Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]);
        let batch_sizes = [3, 3, 1, 1];
        let (mut packed_out, _) = packed_tensor_from_offset(&packed, &batch_sizes, 1);
        // Sequences: [1, 2, 3], [11], [101]
        let _ = packed_out.neg_();
        assert_eq!(
            packed,
            Tensor::of_slice(&[0, 10, 100, -1, -11, -101, -2, -3])
        );
    }

    #[test]
    fn trim_end_n1() {
        // Sequences: [0, 1, 2, 3], [10, 11], [100, 101]
        let packed = Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]);
        let batch_sizes = [3, 3, 1, 1];
        let (packed_out, batch_sizes_out) = packed_tensor_trim_end(&packed, &batch_sizes, 1);
        // Sequences: [0, 1, 2], [10], [100]
        assert_eq!(packed_out, Tensor::of_slice(&[0, 10, 100, 1, 2]));
        assert_eq!(batch_sizes_out, &[3, 1, 1]);
    }

    #[test]
    fn trim_end_is_copy() {
        // Sequences: [0, 1, 2, 3], [10, 11], [100, 101]
        let packed = Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]);
        let batch_sizes = [3, 3, 1, 1];
        let (mut packed_out, _) = packed_tensor_trim_end(&packed, &batch_sizes, 1);
        // Sequences: [0, 1, 2], [10], [100]
        let _ = packed_out.neg_();
        assert_eq!(packed, Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]));
    }

    #[test]
    fn push_shift() {
        // Sequences: [0, 1, 2, 3], [10, 11], [100, 101]
        let packed = Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]);
        let batch_sizes = [3, 3, 1, 1];
        let packed_out = packed_tensor_push_shift(&packed, &batch_sizes, -1);
        // Sequences: [1, 2, 3, -1], [11, -1], [101, -1]
        assert_eq!(
            packed_out,
            Tensor::of_slice(&[1, 11, 101, 2, -1, -1, 3, -1])
        );
    }

    #[test]
    fn packed_tensor_push_shift_is_copy() {
        // Sequences: [0, 1, 2, 3], [10, 11], [100, 101]
        let packed = Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]);
        let batch_sizes = [3, 3, 1, 1];
        let mut packed_out = packed_tensor_push_shift(&packed, &batch_sizes, -1);
        let _ = packed_out.neg_();
        assert_eq!(packed, Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3]));
    }

    #[test]
    fn discounted_cumsum_from_end() {
        // Sequences: [1, 2, 3, 4], [5, 6], [7, 8]
        let packed = Tensor::of_slice(&[1.0, 5.0, 7.0, 2.0, 6.0, 8.0, 3.0, 4.0]);
        let batch_sizes = [3, 3, 1, 1];
        let cumsum = packed_tensor_discounted_cumsum_from_end(&packed, &batch_sizes, 0.1);
        // Sequences: [1.234, 2.34, 3.4, 4], [5.6, 6], [7.8, 8]
        let expected = Tensor::of_slice(&[1.234, 5.6, 7.8, 2.34, 6.0, 8.0, 3.4, 4.0]);
        assert!(
            bool::from(cumsum.isclose(&expected, 1e-8, 1e-8, false).all()),
            "result: {:?}\nexpected: {:?}",
            cumsum,
            expected,
        );
    }
}
