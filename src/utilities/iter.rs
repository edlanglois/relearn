//! Iterator utilities.
use std::cmp::{Ordering, PartialOrd};
use std::error::Error;
use std::fmt;

/// Maximum of a collection of items where the maximum might not exist.
pub trait PartialMax {
    type Item;

    /// A maximum element of an iterator when one exists.
    ///
    /// If several elements are equally maximum then the last one is returned.
    /// Returns a PartialMaxError if there are no elements or the elements are not comparable.
    fn partial_max(self) -> Result<Self::Item, PartialMaxError>;
}

impl<T, I> PartialMax for I
where
    T: PartialOrd,
    I: Iterator<Item = T>,
{
    type Item = T;

    fn partial_max(mut self: I) -> Result<Self::Item, PartialMaxError> {
        match self.try_fold(None, |acc: Option<T>, x| match acc {
            None => Ok(Some(x)),
            Some(a) => match a.partial_cmp(&x) {
                None => Err(()),
                Some(Ordering::Greater) => Ok(Some(a)),
                _ => Ok(Some(x)),
            },
        }) {
            Ok(Some(x)) => Ok(x),
            Ok(None) => Err(PartialMaxError::Empty),
            Err(_) => Err(PartialMaxError::Incomparable),
        }
    }
}

/// Reason that the maximum does not exist.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PartialMaxError {
    /// The collection is empty, there is no maximum.
    Empty,
    /// Some pair of elements cannot be compared.
    Incomparable,
}

impl fmt::Display for PartialMaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PartialMaxError::Empty => "Empty",
                PartialMaxError::Incomparable => "Incomparable",
            }
        )
    }
}

impl Error for PartialMaxError {}

#[cfg(test)]
mod partial_max {
    use super::*;
    use std::f64;

    #[test]
    fn iter_float() {
        assert_eq!(vec![0.0, 3.2, -5.0].into_iter().partial_max(), Ok(3.2));
    }

    #[test]
    fn iter_float_inf() {
        assert_eq!(
            vec![0.0, f64::INFINITY, f64::NEG_INFINITY]
                .into_iter()
                .partial_max(),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn iter_float_nan() {
        assert_eq!(
            vec![0.0, f64::NAN, -5.0].into_iter().partial_max(),
            Err(PartialMaxError::Incomparable)
        );
    }

    #[test]
    fn iter_float_empty() {
        assert_eq!(
            Vec::<f64>::new().into_iter().partial_max(),
            Err(PartialMaxError::Empty)
        );
    }

    #[test]
    fn iter_float_one_nan() {
        // One nan involves no comparison so it is allowed as the max
        assert!(vec![f64::NAN].into_iter().partial_max().is_ok());
    }
}
