//! Ordering utilities
use std::cmp::{Ordering, PartialOrd};

/// Implements a total order for a partial order by panicking whenever elements cannot be compared.
///
/// Note that [`ExpectOrd::partial_cmp`] will panic rather than return None.
/// This is to ensure consistency between the [`PartialOrd`] methods
/// and the [`Ord`] methods, as required by `Ord`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExpectOrd<T> {
    value: T,
}

impl<T> ExpectOrd<T> {
    pub const fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: PartialOrd> PartialEq for ExpectOrd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: PartialOrd> PartialOrd for ExpectOrd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Eq for ExpectOrd<T> {}

impl<T: PartialOrd> Ord for ExpectOrd<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .expect("Cannot be compared")
    }
}

impl<T> From<T> for ExpectOrd<T> {
    fn from(value: T) -> Self {
        Self { value }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod expect_ord {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(1.0, 2.0, Ordering::Less)]
    #[case(1.0, 1.0, Ordering::Equal)]
    #[case(2.0, 1.0, Ordering::Greater)]
    #[case(0.0, f64::INFINITY, Ordering::Less)]
    #[case(f64::INFINITY, f64::NEG_INFINITY, Ordering::Greater)]
    fn float_cmp(#[case] a: f64, #[case] b: f64, #[case] expected: Ordering) {
        assert_eq!(ExpectOrd::from(a).cmp(&ExpectOrd::from(b)), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_cmp_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::from(a).cmp(&ExpectOrd::from(b));
    }

    #[test]
    fn float_sort() {
        let mut v: Vec<_> = vec![5.0, f64::INFINITY, 0.0, -2.5]
            .into_iter()
            .map(ExpectOrd::from)
            .collect();
        v.sort();
        let w: Vec<_> = v.into_iter().map(|x| x.value).collect();
        assert_eq!(w, vec![-2.5, 0.0, 5.0, f64::INFINITY]);
    }

    #[test]
    #[should_panic]
    fn float_sort_nan() {
        let mut v: Vec<_> = vec![0.0, f64::NAN]
            .into_iter()
            .map(ExpectOrd::from)
            .collect();
        v.sort();
    }

    #[rstest]
    #[case(1.0, 2.0, true)]
    #[case(1.0, 1.0, false)]
    #[case(2.0, 1.0, false)]
    fn float_lt(#[case] a: f64, #[case] b: f64, #[case] expected: bool) {
        assert_eq!(ExpectOrd::new(a) < ExpectOrd::new(b), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_lt_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a) < ExpectOrd::new(b);
    }

    #[rstest]
    #[case(1.0, 2.0, true)]
    #[case(1.0, 1.0, true)]
    #[case(2.0, 1.0, false)]
    fn float_le(#[case] a: f64, #[case] b: f64, #[case] expected: bool) {
        assert_eq!(ExpectOrd::new(a) <= ExpectOrd::new(b), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_le_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a) <= ExpectOrd::new(b);
    }

    #[rstest]
    #[case(1.0, 2.0, false)]
    #[case(1.0, 1.0, false)]
    #[case(2.0, 1.0, true)]
    fn float_gt(#[case] a: f64, #[case] b: f64, #[case] expected: bool) {
        assert_eq!(ExpectOrd::new(a) > ExpectOrd::new(b), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_gt_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a) > ExpectOrd::new(b);
    }

    #[rstest]
    #[case(1.0, 2.0, false)]
    #[case(1.0, 1.0, true)]
    #[case(2.0, 1.0, true)]
    fn float_ge(#[case] a: f64, #[case] b: f64, #[case] expected: bool) {
        assert_eq!(ExpectOrd::new(a) >= ExpectOrd::new(b), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_ge_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a) >= ExpectOrd::new(b);
    }

    #[rstest]
    #[case(1.0, 2.0, false)]
    #[case(1.0, 1.0, true)]
    fn float_eq(#[case] a: f64, #[case] b: f64, #[case] expected: bool) {
        assert_eq!(ExpectOrd::new(a) == ExpectOrd::new(b), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_eq_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a) == ExpectOrd::new(b);
    }

    #[rstest]
    #[case(1.0, 2.0, true)]
    #[case(1.0, 1.0, false)]
    fn float_ne(#[case] a: f64, #[case] b: f64, #[case] expected: bool) {
        assert_eq!(ExpectOrd::new(a) != ExpectOrd::new(b), expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_ne_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a) != ExpectOrd::new(b);
    }

    #[rstest]
    #[case(1.0, 2.0, 2.0)]
    #[case(1.0, 0.0, 1.0)]
    #[case(1.0, f64::INFINITY, f64::INFINITY)]
    #[case(1.0, f64::NEG_INFINITY, 1.0)]
    fn float_max(#[case] a: f64, #[case] b: f64, #[case] expected: f64) {
        assert_eq!(ExpectOrd::new(a).max(ExpectOrd::new(b)).value, expected);
    }

    #[rstest]
    #[case(0.0, f64::NAN)]
    #[case(f64::NAN, 1.0)]
    #[case(f64::NAN, f64::NAN)]
    #[should_panic]
    fn float_max_nan(#[case] a: f64, #[case] b: f64) {
        let _ = ExpectOrd::new(a).max(ExpectOrd::new(b));
    }
}
