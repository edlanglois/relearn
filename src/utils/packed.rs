//! Utilities for dealing with packed sequences.
//!
//! A packed set of sequences is an array representing multiple heterogeneous-length sequences
//! where the data is ordered first by offset within a sequence then by sequence index.
//!
//! Sequences must always be packed in order from longest to shortest.
//!
//! For example, the sequences `[0, 1, 2, 3], [10, 11], [100, 101]` are packed as
//! `[0, 10, 100, 1, 11, 101, 2, 3]`.
use std::ops::{Index, Range};

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
#[derive(Debug)]
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
    /// Create a new [PackedIter] instance.
    ///
    /// # Args
    /// * `sequences` - A iterator of sequence iterators.
    ///     The inner sequence iterators must have fixed lengths.
    ///
    /// **Note**: Sequences will be reordered (stably) so that
    /// they are in order of monotonically decreasing length.
    pub fn new<U>(sequences: U) -> Self
    where
        U: IntoIterator,
        <U as IntoIterator>::Item: IntoIterator<IntoIter = T>,
    {
        let mut sequences: Vec<_> = sequences.into_iter().map(|s| s.into_iter()).collect();
        // Sort in descending order of length.
        sequences.sort_by(|a, b| a.len().cmp(&b.len()).reverse());
        let batch_size = sequences.len();
        Self {
            sequences,
            seq_idx: 0,
            batch_size,
        }
    }

    /// Create a new [PackedIter] from sequences sorted in descending order of length.
    ///
    /// # Args
    /// * `sequences` - A iterator of sequence iterators.
    ///     The inner sequence iterators must have fixed lengths
    ///     and be sorted in order of monotonically decreasing length.
    pub fn from_sorted<U>(sequences: U) -> Self
    where
        U: IntoIterator,
        <U as IntoIterator>::Item: IntoIterator<IntoIter = T>,
    {
        let sequences: Vec<_> = sequences.into_iter().map(|s| s.into_iter()).collect();
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
    type Item = <T as Iterator>::Item;

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
#[derive(Debug)]
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
    /// Create a new [PackingIndices] instance from ranges sorted in decreasing order of length.
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
#[derive(Debug)]
pub struct PackedBatchSizes<'a, T> {
    /// Sequences lengths sorted in monotonically decreasing order.
    sequence_lengths: &'a [T],
    /// Current offset within the sequences.
    offset: usize,
    /// Number of sequences with length > offset
    batch_size: usize,
}

impl<'a> PackedBatchSizes<'a, usize> {
    /// Create a new [PackedBatchSizes] instance from monotonically decreasing sequence lengths.
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
    /// Create a new [PackedBatchSizes] instance from ranges with monotonically decreasing lengths.
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

#[cfg(test)]
mod packed_iter {
    use super::*;

    #[test]
    fn iter_from_sorted() {
        let sequences = [vec![0, 1, 2, 3], vec![10, 11], vec![100, 101]];
        let packed: Vec<_> = PackedIter::from_sorted(&sequences).map(|&x| x).collect();
        let expected = vec![0, 10, 100, 1, 11, 101, 2, 3];
        assert_eq!(packed, expected);
    }

    #[test]
    fn iter_from_unsorted() {
        let sequences = [vec![10, 11], vec![0, 1, 2, 3], vec![100, 101]];
        let packed: Vec<_> = PackedIter::new(&sequences).map(|&x| x).collect();
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
