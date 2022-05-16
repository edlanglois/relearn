//! Packed Tensors
use super::serialize::TensorDef;
use crate::torch::tensors::ExclusiveTensor;
use crate::utils::sequence::Sequence;
use ndarray::{azip, ArrayViewMut, Axis, IxDyn, Slice};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::iter;
use std::iter::{Fuse, FusedIterator};
use std::ops::{AddAssign, Bound, Mul};
use std::rc::Rc;
use tch::{kind::Element, Device, IndexOp, Kind, Tensor};
use thiserror::Error;

/// Error involving packing data.
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PackingError {
    #[error("sequences lengths or batch sizes increased; should be monotonic decreasing")]
    Increasing,
    #[error("input tensor has < {expected} dimensions")]
    TooFewDimensions { expected: u8 },
}

/// A packed tensor.
///
/// A packed tensor represents a set of heterogeneous-length sequences.
/// The sequences are arranged along the first dimension of the tensor and are stored interleaved:
/// the first steps from all sequences followed by the second steps, etc.
///
/// The sequences are packed in order from longest to shortest.
///
/// For example, the sequences `[0, 1, 2, 3]`, `[10, 11]`, `[100, 101]` are packed as
/// `[0, 10, 100, 1, 11, 101, 2, 3]`.
#[must_use]
#[serde_as]
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct PackedTensor {
    /// The packed tensor data. Must have at least one dimension.
    #[serde_as(as = "TensorDef")]
    tensor: Tensor,
    /// The packed structure of `tensor`.
    structure: PackedStructure,
}

impl Clone for PackedTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.shallow_clone(),
            structure: self.structure.clone(),
        }
    }
}

impl PackedTensor {
    /// Construct from a packed data [`Tensor`] and a `PackedStructure` describing it.
    ///
    /// # Panics
    /// * If the tensor is 0-dimensional.
    /// * If the length of the first dimension does not match `structure.len()`.
    #[inline]
    pub fn from_parts(tensor: Tensor, structure: PackedStructure) -> Self {
        assert_eq!(
            structure.len() as i64,
            *tensor
                .size()
                .first()
                .expect("tensor must have at least 1 dimension"),
            "structure length does not match tensor first dimension size"
        );
        Self { tensor, structure }
    }

    /// Construct from an aligned tensor with equal-length sequences.
    ///
    /// # Inputs
    /// * `tensor`: A tensor with shape `[SEQUENCE_LEN, NUM_SEQUENCES, ...]`.
    ///
    /// # Returns
    /// Returns an error if the input tensor has less than 2 dimensions.
    pub fn from_aligned_tensor(tensor: &Tensor) -> Result<Self, PackingError> {
        let mut size = tensor.size();
        if size.len() < 2 {
            return Err(PackingError::TooFewDimensions { expected: 2 });
        }
        let sequence_length = size.remove(0);
        let batch_size = size[0];

        size[0] *= sequence_length;
        Ok(Self {
            tensor: tensor.reshape(&size),
            structure: PackedStructure::Aligned {
                sequence_length: sequence_length.try_into().unwrap(),
                batch_size: batch_size.try_into().unwrap(),
            },
        })
    }

    /// Construct a 1D packed tensor from slices sorted in monotonic decreasing order of length.
    ///
    /// Returns an error if any slice is longer than the sequence before it.
    #[inline]
    pub fn from_sorted_sequences<'a, I, E>(slices: I) -> Result<Self, PackingError>
    where
        I: IntoIterator<Item = &'a [E]>,
        I::IntoIter: Clone,
        E: 'a + tch::kind::Element + Copy,
    {
        let sequences = slices.into_iter();
        let structure =
            PackedStructure::from_sorted_sequence_lengths(sequences.clone().map(<[E]>::len))?;
        let data: Vec<_> = PackedSeqIter::from_sorted(sequences).copied().collect();
        let tensor = Tensor::of_slice(&data);
        Ok(Self { tensor, structure })
    }

    /// Convert into the underlying packed [`Tensor`] object.
    #[allow(clippy::missing_const_for_fn)] // false positive
    #[inline]
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Reference the underlying packed [`Tensor`] object.
    #[inline]
    pub const fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Mutably reference the underlying packed [`Tensor`] object.
    #[inline]
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }

    /// Reference the packed structure.
    #[must_use]
    #[inline]
    pub const fn structure(&self) -> &PackedStructure {
        &self.structure
    }

    /// The tensor [`Kind`] (data type).
    #[must_use]
    pub fn kind(&self) -> Kind {
        self.tensor.kind()
    }

    /// The tensor [`Device`].
    #[must_use]
    pub fn device(&self) -> Device {
        self.tensor.device()
    }

    /// A [`Tensor`] with the packed batch sizes.
    ///
    /// Has type `i64` and is on the CPU device.
    pub fn batch_sizes_tensor(&self) -> Tensor {
        self.structure.batch_sizes_tensor()
    }

    /// Batch size of the first step if any. The largest batch size.
    #[must_use]
    pub fn first_batch_size(&self) -> Option<i64> {
        self.structure.first_batch_size()
    }

    /// Transform the stored tensor with a function that preserves the packed sequence structure.
    ///
    /// The sequence structure is stored along the first dimension of the tensor. This dimension
    /// must be preserved; its length must not change and its semantics should be preserved.
    /// Other methods may panic if the function changes the sequence structure.
    #[inline]
    pub fn batch_map<F: FnOnce(Tensor) -> Tensor>(self, f: F) -> Self {
        Self {
            tensor: f(self.tensor),
            structure: self.structure,
        }
    }

    /// Map the stored tensor with a function that preserves the packed sequence structure.
    ///
    /// The sequence structure is stored along the first dimension of the tensor. This dimension
    /// must be preserved; its length must not change and its semantics should be preserved.
    /// Other methods may panic if the function changes the sequence structure.
    #[inline]
    pub fn batch_map_ref<'a, F: FnOnce(&'a Tensor) -> Tensor>(&'a self, f: F) -> Self {
        Self {
            tensor: f(&self.tensor),
            structure: self.structure.clone(),
        }
    }

    /// View the packed tensor with the first `n` items removed from each sequence.
    pub fn view_trim_start(&self, n: usize) -> Self {
        let (to_remove, structure) = match &self.structure {
            PackedStructure::Aligned {
                sequence_length,
                batch_size,
            } => {
                let n = n.min(*sequence_length);
                let to_remove = n * *batch_size;
                let new_structure = PackedStructure::Aligned {
                    sequence_length: *sequence_length - n,
                    batch_size: *batch_size,
                };
                (to_remove as i64, new_structure)
            }
            PackedStructure::Ragged(batch_sizes) => {
                let to_remove = batch_sizes.as_slice()[..n].iter().copied().sum();
                let new_structure = PackedStructure::Ragged(batch_sizes.clone().trim(n));
                (to_remove, new_structure)
            }
        };
        let tensor = self.tensor.i(to_remove..);

        Self { tensor, structure }
    }

    /// Copy a packed tensor without the last `n` elements of each sequence.
    ///
    /// Sequences with `n` or fewer elements are removed.
    pub fn trim_end(&self, n: usize) -> Self {
        match &self.structure {
            PackedStructure::Aligned {
                sequence_length,
                batch_size,
            } => {
                let n = n.min(*sequence_length);
                let tensor = self.tensor.i(..(n * *batch_size) as i64);
                let structure = PackedStructure::Aligned {
                    sequence_length: *sequence_length - n,
                    batch_size: *batch_size,
                };
                Self { tensor, structure }
            }
            PackedStructure::Ragged(batch_sizes) => {
                let new_batch_sizes = batch_sizes.clone().trim(n);
                let (old_group_sizes, new_group_sizes): (Vec<_>, Vec<_>) =
                    GroupBatchesForResize::new(
                        batch_sizes.as_slice().iter().copied(),
                        new_batch_sizes.as_slice().iter().copied(),
                    )
                    .unzip();

                // Split the tensor into groups based on the old sizes
                let groups = self.tensor.split_with_sizes(&old_group_sizes, 0);

                // Resize each group into the new group size
                // If batch_sizes is monotonic decreasing (which it should be) then each new group
                // size will be less than or equal to the old group size.
                let new_groups: Vec<_> = groups
                    .iter()
                    .zip(new_group_sizes)
                    .map(|(group, new_size)| group.i(..new_size))
                    .collect();

                // Collect the groups back into a single tensor
                let new_tensor = Tensor::cat(&new_groups, 0);

                Self {
                    tensor: new_tensor,
                    structure: PackedStructure::Ragged(new_batch_sizes),
                }
            }
        }
    }

    /// Discounted cumulative sum from sequence end to start for a tensor.
    ///
    /// For each element `x[i]` in the sequence `x[0] ... x[N]`,
    /// returns `y[i] = sum_{j in i..N} discount ** (j - i) * x[j]`
    ///
    /// # Warning
    /// Does not preserve gradients.
    ///
    /// # Panics
    /// If the type of `discount` does not match the tensor data type.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn discounted_cumsum_from_end<T>(&self, discount: T) -> Self
    where
        T: Mul + AddAssign<<T as Mul>::Output> + Copy + Element,
    {
        let mut out = ExclusiveTensor::<T, _>::try_copy_from(self.tensor()).unwrap();
        match &self.structure {
            PackedStructure::Ragged(batch_sizes) => {
                inplace_discounted_cumsum_from_end(
                    out.array_view_mut(),
                    discount,
                    batch_sizes.as_slice().iter().map(|b| *b as usize).rev(),
                );
            }
            PackedStructure::Aligned {
                sequence_length,
                batch_size,
            } => {
                inplace_discounted_cumsum_from_end(
                    out.array_view_mut(),
                    discount,
                    iter::repeat(*batch_size).take(*sequence_length),
                );
            }
        }
        Self {
            tensor: out.into_tensor().to_device(self.tensor.device()),
            structure: self.structure.clone(),
        }
    }
}

#[allow(clippy::cast_possible_wrap)]
fn inplace_discounted_cumsum_from_end<I, T>(
    mut array: ArrayViewMut<T, IxDyn>,
    discount: T,
    rev_batch_sizes: I, // Batch sizes in reverse order
) where
    I: IntoIterator<Item = usize>,
    T: Mul + AddAssign<<T as Mul>::Output> + Copy,
{
    // Everything to the right of offset is complete, to the left is incomplete
    let mut offset = array.shape()[0]; // Panics if array is 0-dimensional
    for batch_size in rev_batch_sizes {
        let (left, prev_batch) = array.split_at(Axis(0), offset);
        array = left;
        offset -= batch_size;

        let prev_batch_size = prev_batch.shape()[0];
        let batch_part = array.slice_axis_mut(
            Axis(0),
            Slice {
                start: offset as isize,
                end: Some((offset + prev_batch_size) as isize),
                step: 1,
            },
        );
        azip!((a in batch_part, b in &prev_batch) *a += *b * discount);
    }
    assert_eq!(
        offset, 0,
        "batch sizes do not match array first dimension length"
    );
}

/// Information about a packed tensor structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PackedStructure {
    /// Heterogeneous batch sizes lengths
    Ragged(SharedBatchSizes),
    /// All sequences have the same length.
    Aligned {
        sequence_length: usize,
        /// Number of sequences
        batch_size: usize,
    },
}

impl PackedStructure {
    /// Construct from an iterator of monotonic decreasing batch sizes.
    ///
    /// Returns an error if any batch size is greater than the previous batch size.
    pub fn from_batch_sizes<I: IntoIterator<Item = usize>>(
        batch_sizes: I,
    ) -> Result<Self, PackingError> {
        Ok(Self::Ragged(SharedBatchSizes::from_batch_sizes(
            batch_sizes,
        )?))
    }

    /// Construct from an iterator of monotonic decreasing sequence lengths.
    ///
    /// Returns an error if any length is greater than the previous length.
    pub fn from_sorted_sequence_lengths<I: IntoIterator<Item = usize>>(
        lengths: I,
    ) -> Result<Self, PackingError> {
        Ok(Self::Ragged(
            SharedBatchSizes::from_sorted_sequence_lengths(lengths)?,
        ))
    }

    /// A [`Tensor`] with the packed batch sizes.
    ///
    /// Has type `i64` and is on the CPU device.
    pub fn batch_sizes_tensor(&self) -> Tensor {
        match self {
            Self::Ragged(batch_sizes) => batch_sizes.tensor(),
            Self::Aligned {
                sequence_length,
                batch_size,
            } => Tensor::full(
                &[*sequence_length as i64],
                *batch_size as i64,
                (Kind::Int64, Device::Cpu),
            ),
        }
    }

    /// Batch size of the first step if any. The largest batch size.
    #[must_use]
    pub fn first_batch_size(&self) -> Option<i64> {
        match self {
            Self::Ragged(batch_sizes) => batch_sizes.as_slice().first().copied(),
            Self::Aligned {
                sequence_length,
                batch_size,
            } => {
                if *sequence_length > 0 {
                    Some(*batch_size as _)
                } else {
                    None
                }
            }
        }
    }

    /// The total number of elements across all sequences represented by this structure.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Ragged(batch_sizes) => batch_sizes.len(),
            Self::Aligned {
                sequence_length,
                batch_size,
            } => sequence_length * batch_size,
        }
    }

    /// Whether the total number of elements across all sequences is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Ragged(batch_sizes) => batch_sizes.is_empty(),
            Self::Aligned {
                sequence_length,
                batch_size,
            } => *sequence_length == 0 || *batch_size == 0,
        }
    }

    /// Structure resulting from removing `n` items from each sequence.
    ///
    /// Any sequences with length less than `n` are reduced to length `0`.
    #[allow(clippy::missing_const_for_fn)] // false positive
    #[must_use]
    pub fn trim(self, n: usize) -> Self {
        match self {
            Self::Ragged(batch_sizes) => Self::Ragged(batch_sizes.trim(n)),
            Self::Aligned {
                sequence_length,
                batch_size,
            } => Self::Aligned {
                sequence_length: sequence_length.saturating_sub(n),
                batch_size,
            },
        }
    }
}

/// Slice of a reference-counted batch sizes vector.
///
/// The value of the `i`th index of the batch sizes vector is the number of sequences with length
/// at least `i + 1`. It is the number of index-`i` steps that appear in the packed tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedBatchSizes {
    root: Rc<BatchSizes>,
    start: usize,       // inclusive
    end: Option<usize>, // exclusive
}

impl AsRef<[i64]> for SharedBatchSizes {
    #[inline]
    fn as_ref(&self) -> &[i64] {
        self.as_slice()
    }
}

impl<T: AsRef<[i64]>> PartialEq<T> for SharedBatchSizes {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for SharedBatchSizes {}

impl SharedBatchSizes {
    /// Construct from an iterator of monotonic decreasing batch sizes.
    ///
    /// Returns an error if any batch size is greater than the previous batch size.
    pub fn from_batch_sizes<I: IntoIterator<Item = usize>>(
        batch_sizes: I,
    ) -> Result<Self, PackingError> {
        Ok(Self {
            root: Rc::new(BatchSizes::from_batch_sizes(batch_sizes)?),
            start: 0,
            end: None,
        })
    }

    /// Construct from an iterator of monotonic decreasing sequence lengths.
    ///
    /// Returns an error if any length is greater than the previous length.
    pub fn from_sorted_sequence_lengths<I: IntoIterator<Item = usize>>(
        lengths: I,
    ) -> Result<Self, PackingError> {
        Ok(Self {
            root: Rc::new(BatchSizes::from_sorted_sequence_lengths(lengths)?),
            start: 0,
            end: None,
        })
    }

    /// View batch sizes as a slice
    #[inline]
    pub fn as_slice(&self) -> &[i64] {
        let start = Bound::Included(self.start);
        let end = self.end.map_or(Bound::Unbounded, Bound::Excluded);
        &self.root.as_slice()[(start, end)]
    }

    /// Batch size as an `i64` `[Tensor]`. The underlying data is cached and shared between calls.
    #[inline]
    pub fn tensor(&self) -> Tensor {
        let root_tensor = self.root.as_tensor();

        if self.start == 0 && self.end.is_none() {
            root_tensor.shallow_clone()
        } else {
            let end = self.end.map(|i| i as i64);
            root_tensor.slice(0, self.start as i64, end, 1)
        }
    }

    /// The total number of elements across all sequences represented by this structure.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice()
            .iter()
            // Batch sizes are initalized from usize and should still fit
            .map(|x| usize::try_from(*x).unwrap())
            .sum()
    }

    /// Whether the total number of elements across all sequences is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_slice().iter().all(|x| *x == 0)
    }

    /// Batch sizes resulting from removing `n` values from each sequence.
    #[must_use]
    pub const fn trim(mut self, n: usize) -> Self {
        self.start += n;
        self
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchSizes {
    /// Batch sizes. Non-negative and monotonic decreasing.
    ///
    /// These would be `usize` except `tch` expects `i64`.
    batch_sizes: Vec<i64>,

    /// Cached `batch_sizes` as a non-negative i64 tensor on the CPU device.
    #[serde(skip)]
    batch_sizes_tensor: OnceCell<Tensor>,
}

impl AsRef<[i64]> for BatchSizes {
    #[inline]
    fn as_ref(&self) -> &[i64] {
        self.as_slice()
    }
}

impl BatchSizes {
    /// Construct from an iterator of monotonic decreasing batch sizes.
    ///
    /// Returns an error if any batch size is greater than the previous batch size.
    pub fn from_batch_sizes<I: IntoIterator<Item = usize>>(
        batch_sizes: I,
    ) -> Result<Self, PackingError> {
        let mut prev = usize::MAX;
        let batch_sizes: Vec<_> = batch_sizes
            .into_iter()
            .map(|x| {
                if x > prev {
                    Err(PackingError::Increasing)
                } else {
                    prev = x;
                    Ok(x as i64)
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            batch_sizes,
            batch_sizes_tensor: OnceCell::new(),
        })
    }

    /// Construct from an iterator of monotonic decreasing sequence lengths.
    ///
    /// Returns an an error if any length is greater than the previous length.
    pub fn from_sorted_sequence_lengths<I: IntoIterator<Item = usize>>(
        lengths: I,
    ) -> Result<Self, PackingError> {
        let mut lengths = lengths.into_iter().enumerate().peekable();

        let (_, max_seq_len) = lengths.peek().copied().unwrap_or((0, 0));
        let mut batch_sizes = vec![0; max_seq_len];

        while let Some((i, seq_len)) = lengths.next() {
            // `batch_size = i + 1` sequences have length at least `seq_len`.
            // Record this as the batch size for all lengths down to the length of the next seq
            let (_, next_len) = lengths.peek().copied().unwrap_or((0, 0));
            if next_len > seq_len {
                return Err(PackingError::Increasing);
            }
            batch_sizes[next_len..seq_len].fill((i + 1) as i64);
        }
        Ok(Self {
            batch_sizes,
            batch_sizes_tensor: OnceCell::new(),
        })
    }

    /// View batch sizes as a slice
    #[inline]
    pub fn as_slice(&self) -> &[i64] {
        self.batch_sizes.as_slice()
    }

    /// View batch sizes as an `i64` `[Tensor]` (cached).
    #[inline]
    pub fn as_tensor(&self) -> &Tensor {
        self.batch_sizes_tensor
            .get_or_init(|| Tensor::of_slice(&self.batch_sizes))
    }

    /// The total number of elements across all sequences represented by this structure.
    #[inline]
    pub fn len(&self) -> usize {
        self.batch_sizes
            .iter()
            // Batch sizes are initalized from usize and should still fit
            .map(|x| usize::try_from(*x).unwrap())
            .sum()
    }

    /// Whether the total number of elements across all sequences is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch_sizes.iter().all(|x| *x == 0)
    }
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
/// # Items
/// Yields pairs `(old_group_size, new_group_size)` each representing a group of elements to be
/// resized only at the end of the group.
struct GroupBatchesForResize<A, B> {
    old_batch_sizes: Fuse<A>,
    new_batch_sizes: Fuse<B>,
}

impl<A, B> GroupBatchesForResize<A, B>
where
    A: Iterator,
    B: Iterator,
{
    pub fn new<IA, IB>(old_batch_sizes: IA, new_batch_sizes: IB) -> Self
    where
        IA: IntoIterator<IntoIter = A>,
        IB: IntoIterator<IntoIter = B>,
    {
        Self {
            old_batch_sizes: old_batch_sizes.into_iter().fuse(),
            new_batch_sizes: new_batch_sizes.into_iter().fuse(),
        }
    }
}

impl<A, B> Iterator for GroupBatchesForResize<A, B>
where
    A: Iterator<Item = i64>,
    B: Iterator<Item = i64>,
{
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        // Accumulated old/new group sizes.
        //
        // Invariant: These have the same size unless one iterator has ended.
        let mut old_group_size = 0;
        let mut new_group_size = 0;
        loop {
            let (old, new, tail) = match (self.old_batch_sizes.next(), self.new_batch_sizes.next())
            {
                (Some(old), Some(new)) => (old, new, false),
                (Some(old), None) => (old, 0, true),
                (None, Some(new)) => (0, new, true),
                (None, None) => break,
            };
            old_group_size += old;
            new_group_size += new;

            // Return the group if the sizes differ.
            // Can merge consecutive tail batches because the whole tail will be added/removed
            // so it is still a suffix operation on the group, just one that affects multiple
            // batches.
            if !tail && old != new {
                break;
            }
        }
        if (old_group_size, new_group_size) == (0, 0) {
            None
        } else {
            Some((old_group_size, new_group_size))
        }
    }
}

impl<A, B> FusedIterator for GroupBatchesForResize<A, B>
where
    A: Iterator<Item = i64>,
    B: Iterator<Item = i64>,
{
}

/// Iterator that packs together the elements of multiple sequences.
///
/// Does not allocate any heap memory.
///
/// # Example
/// ```
/// use relearn::torch::packed::PackedSeqIter;
///
/// let sequences: [&[_]; 3] = [&[0, 1, 2, 3], &[10, 11], &[100, 101]];
/// let packed: Vec<_> = PackedSeqIter::from_sorted(&sequences).copied().collect();
/// assert_eq!(packed, vec![0, 10, 100, 1, 11, 101, 2, 3]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedSeqIter<I> {
    /// Initial copy of the sequence iterator. Never modified.
    sequences: I,

    /// Current offset within the sequences.
    offset: usize,
    /// Iterator of sequences for the current offset.
    sequences_iter: I,
}

impl<I> PackedSeqIter<I>
where
    I: Iterator + Clone,
    <I as Iterator>::Item: Sequence,
{
    /// Initialize from sequences sorted in monotonic decreasing order of length.
    pub fn from_sorted<T: IntoIterator<IntoIter = I>>(into_sequences: T) -> Self {
        let sequences = into_sequences.into_iter();
        assert!(
            sequences
                .clone()
                .zip(sequences.clone().skip(1))
                .all(|(a, b)| a.len() >= b.len()),
            "sequences not in monotonic decreasing order of length"
        );
        let sequences_iter = sequences.clone();
        Self {
            sequences,
            offset: 0,
            sequences_iter,
        }
    }
}

impl<I> Iterator for PackedSeqIter<I>
where
    I: Iterator + Clone,
    <I as Iterator>::Item: Sequence,
{
    type Item = <I::Item as Sequence>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self
            .sequences_iter
            .next()
            .and_then(|seq| seq.get(self.offset))
        {
            Some(value)
        } else {
            // Increment offset and restart the loop
            self.offset += 1;
            self.sequences_iter = self.sequences.clone();
            // If this fails then there are no more items left
            self.sequences_iter
                .next()
                .and_then(|seq| seq.get(self.offset))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Total size of all elements at index `offset` or later.
        let level_size: usize = self
            .sequences
            .clone()
            .map(|seq| seq.len().saturating_sub(self.offset))
            .take_while(|&size| size > 0)
            .sum();
        let size = if level_size == 0 {
            // This is handled because sequences_iter ends up incorrectly indicating that one
            // element has been emitted when the iterator has been fully exhausted.
            0
        } else {
            // Subtract the number of elements emitted so far at this offset level.
            // NOTE: Could use size_hint() for these but already iterating over sequences to
            // calculate level_size.
            level_size - (self.sequences.clone().count() - self.sequences_iter.clone().count())
        };
        (size, Some(size))
    }
}

impl<I> ExactSizeIterator for PackedSeqIter<I>
where
    I: ExactSizeIterator + Clone,
    <I as Iterator>::Item: Sequence,
{
}

#[cfg(test)]
mod packed_seq_iter {
    use super::*;

    #[test]
    fn iter() {
        let data = [0, 1, 2, 3, 10, 11, 100, 101];
        let ranges = [0..4, 4..6, 6..8];
        let packed: Vec<_> = PackedSeqIter::from_sorted(&ranges)
            .map(|i| data[i])
            .collect();
        let expected = vec![0, 10, 100, 1, 11, 101, 2, 3];
        assert_eq!(packed, expected);
    }

    #[test]
    fn size_hint() {
        let ranges = [0..4, 4..6, 6..8];
        let packing_indices = PackedSeqIter::from_sorted(&ranges);
        assert_eq!(packing_indices.size_hint(), (8, Some(8)));
    }

    #[test]
    fn size_hint_after_next() {
        let ranges = [0..4, 4..6, 6..8];
        let mut packing_indices = PackedSeqIter::from_sorted(&ranges);
        let _ = packing_indices.next();
        assert_eq!(packing_indices.size_hint(), (7, Some(7)));
        let _ = packing_indices.next();
        assert_eq!(packing_indices.size_hint(), (6, Some(6)));
    }
}

#[cfg(test)]
mod batch_sizes {
    use super::*;

    #[test]
    fn from_sorted() {
        let batch_sizes = BatchSizes::from_sorted_sequence_lengths([4, 2, 2]).unwrap();
        assert_eq!(batch_sizes.batch_sizes, [3, 3, 1, 1]);
    }

    #[test]
    fn from_increasing() {
        assert_eq!(
            BatchSizes::from_sorted_sequence_lengths([4, 5, 2]).unwrap_err(),
            PackingError::Increasing
        );
    }
}

#[cfg(test)]
#[allow(clippy::needless_pass_by_value)]
mod packed_tensor {
    use super::*;
    use rstest::{fixture, rstest};

    /// Packed tensor representing sequences: `[0, 1, 2, 3]`, `[10, 11]`, and `[100, 101]`.
    #[fixture]
    fn packed_tensor() -> PackedTensor {
        PackedTensor::from_sorted_sequences([&[0, 1, 2, 3] as &[_], &[10, 11], &[100, 101]])
            .unwrap()
    }

    #[test]
    fn from_sorted_sequences() {
        let packed_tensor =
            PackedTensor::from_sorted_sequences([&[0, 1, 2, 3] as &[_], &[10, 11], &[100, 101]])
                .unwrap();
        assert_eq!(
            packed_tensor.tensor(),
            &Tensor::of_slice(&[0, 10, 100, 1, 11, 101, 2, 3])
        );
        assert_eq!(
            packed_tensor.batch_sizes_tensor(),
            Tensor::of_slice(&[3, 3, 1, 1])
        );
    }

    #[rstest]
    fn view_trim_start_n1(packed_tensor: PackedTensor) {
        let actual = packed_tensor.view_trim_start(1);
        let expected =
            PackedTensor::from_sorted_sequences([&[1, 2, 3] as &[_], &[11], &[101]]).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn view_trim_start_n3(packed_tensor: PackedTensor) {
        let actual = packed_tensor.view_trim_start(3);
        // Sequences: [3]
        let expected = PackedTensor::from_sorted_sequences([&[3] as &[_]]).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn view_trim_start_is_view(packed_tensor: PackedTensor) {
        let mut trimmed = packed_tensor.view_trim_start(1);
        let _ = trimmed.tensor.neg_();

        let expected = PackedTensor::from_sorted_sequences([
            &[0, -1, -2, -3] as &[_],
            &[10, -11],
            &[100, -101],
        ])
        .unwrap();
        assert_eq!(packed_tensor, expected);
    }

    #[rstest]
    fn trim_end_n1(packed_tensor: PackedTensor) {
        let actual = packed_tensor.trim_end(1);
        let expected =
            PackedTensor::from_sorted_sequences([&[0, 1, 2] as &[_], &[10], &[100]]).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn trim_end_n3(packed_tensor: PackedTensor) {
        let actual = packed_tensor.trim_end(3);
        let expected = PackedTensor::from_sorted_sequences([&[0] as &[_]]).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn trim_end_is_copy(packed_tensor: PackedTensor) {
        let mut trimmed = packed_tensor.trim_end(1);
        let _ = trimmed.tensor.neg_();

        // packed_tensor is unchanged
        let expected =
            PackedTensor::from_sorted_sequences([&[0, 1, 2, 3] as &[_], &[10, 11], &[100, 101]])
                .unwrap();
        assert_eq!(packed_tensor, expected);
    }

    #[test]
    fn discounted_cumsum_from_end() {
        let packed_tensor = PackedTensor::from_sorted_sequences([
            &[1.0, 2.0, 3.0, 4.0] as &[_],
            &[5.0, 6.0],
            &[7.0, 8.0],
        ])
        .unwrap();

        let cumsum = packed_tensor.discounted_cumsum_from_end(0.1);

        // Sequences: [1.234, 2.34, 3.4, 4], [5.6, 6], [7.8, 8]
        let expected = PackedTensor::from_sorted_sequences([
            &[1.234, 2.34, 3.4, 4.0] as &[_],
            &[5.6, 6.0],
            &[7.8, 8.0],
        ])
        .unwrap();
        assert_eq!(cumsum.structure, expected.structure);
        assert!(
            bool::from(
                cumsum
                    .tensor
                    .isclose(&expected.tensor, 1e-8, 1e-8, false)
                    .all()
            ),
            "result: {:?}\nexpected: {:?}",
            cumsum,
            expected,
        );
    }

    #[rstest]
    fn batch_sizes_tensor_values(packed_tensor: PackedTensor) {
        let actual = packed_tensor.structure.batch_sizes_tensor();
        let expected = Tensor::of_slice(&[3, 3, 1, 1]);
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn batch_sizes_tensor_device_cpu(packed_tensor: PackedTensor) {
        let batch_sizes = packed_tensor.structure.batch_sizes_tensor();
        assert_eq!(batch_sizes.device(), tch::Device::Cpu);
    }
}
