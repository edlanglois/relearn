//! Mathematical finite-dimensional vector.
use num_traits::{MulAdd, MulAddAssign, Zero};
use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A mathematical coordinate vector in a finite dimensional vector space.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CoordVector<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> From<[T; N]> for CoordVector<T, N> {
    fn from(coordinates: [T; N]) -> Self {
        Self(coordinates)
    }
}

impl<T: Zero, const N: usize> Default for CoordVector<T, N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T, const N: usize> Deref for CoordVector<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for CoordVector<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Zero, const N: usize> Zero for CoordVector<T, N> {
    fn zero() -> Self {
        Self(array_init::array_init(|_| T::zero()))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(T::is_zero)
    }

    fn set_zero(&mut self) {
        self.0.iter_mut().for_each(T::set_zero)
    }
}

/// Vector addition
impl<T: Add<Output = T>, const N: usize> Add for CoordVector<T, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(array_init::from_iter(self.0.into_iter().zip(other.0).map(|(a, b)| a + b)).unwrap())
    }
}

/// In-place vector addition
impl<T: AddAssign, const N: usize> AddAssign for CoordVector<T, N> {
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a += b
        }
    }
}

/// Vector subtraction
impl<T: Sub<Output = T>, const N: usize> Sub for CoordVector<T, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(array_init::from_iter(self.0.into_iter().zip(other.0).map(|(a, b)| a - b)).unwrap())
    }
}

/// In-place vector subtraction
impl<T: SubAssign, const N: usize> SubAssign for CoordVector<T, N> {
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a -= b
        }
    }
}

/// Negation
impl<T: Neg<Output = T>, const N: usize> Neg for CoordVector<T, N> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(array_init::from_iter(self.0.map(|a| -a)).unwrap())
    }
}

/// Scalar multiplication
impl<T: Mul<Output = T> + Copy, const N: usize> Mul<T> for CoordVector<T, N> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self(array_init::from_iter(self.0.map(|a| a * scalar)).unwrap())
    }
}

/// In-place scalar multiplication
impl<T: MulAssign + Copy, const N: usize> MulAssign<T> for CoordVector<T, N> {
    fn mul_assign(&mut self, scalar: T) {
        for a in self.0.iter_mut() {
            *a *= scalar;
        }
    }
}

/// Scalar division
impl<T: Div<Output = T> + Copy, const N: usize> Div<T> for CoordVector<T, N> {
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self(array_init::from_iter(self.0.map(|a| a / scalar)).unwrap())
    }
}

/// In-place scalar division
impl<T: DivAssign + Copy, const N: usize> DivAssign<T> for CoordVector<T, N> {
    fn div_assign(&mut self, scalar: T) {
        for a in self.0.iter_mut() {
            *a /= scalar;
        }
    }
}

/// Fused scalar multiply and vector add.
impl<T: MulAdd<Output = T> + Copy, const N: usize> MulAdd<T> for CoordVector<T, N> {
    type Output = Self;

    fn mul_add(self, scalar: T, vector: Self) -> Self {
        Self(
            array_init::from_iter(
                self.0
                    .into_iter()
                    .zip(vector.0)
                    .map(|(a, b)| a.mul_add(scalar, b)),
            )
            .unwrap(),
        )
    }
}

/// In-place fused scalar multiply and vector add.
impl<T: MulAddAssign + Copy, const N: usize> MulAddAssign<T> for CoordVector<T, N> {
    fn mul_add_assign(&mut self, scalar: T, vector: Self) {
        for (a, b) in self.0.iter_mut().zip(vector.0) {
            a.mul_add_assign(scalar, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_array() {
        let v = CoordVector::from([1, 2, 3]);
        assert_eq!(CoordVector([1, 2, 3]), v);
    }

    #[test]
    fn default() {
        let v: CoordVector<i64, 3> = CoordVector::default();
        assert_eq!(CoordVector([0, 0, 0]), v);
    }

    #[test]
    fn deref() {
        let v = CoordVector([1, 2, 3]);
        assert_eq!(v.len(), 3);
        let v_slice: &[isize] = &v;
        assert_eq!(v_slice.len(), 3);
    }

    #[test]
    fn deref_mut() {
        let mut v = CoordVector([1, 2, 3]);
        v.reverse();
        assert_eq!(v, CoordVector([3, 2, 1]));
    }

    #[test]
    fn zero_i64() {
        let v: CoordVector<i64, 3> = CoordVector::zero();
        assert_eq!(CoordVector([0, 0, 0]), v);
    }

    #[test]
    fn zero_f64() {
        let v: CoordVector<f64, 3> = CoordVector::zero();
        assert_eq!(CoordVector([0.0, 0.0, 0.0]), v);
    }

    #[test]
    fn is_zero_true() {
        let v = CoordVector([0, 0, 0]);
        assert!(v.is_zero());
    }

    #[test]
    fn is_zero_false() {
        let v = CoordVector([0, 1, 0]);
        assert!(!v.is_zero());
    }

    #[test]
    fn is_zero_empty() {
        let v = CoordVector::<i64, 0>([]);
        assert!(v.is_zero());
    }

    #[test]
    fn set_zero() {
        let mut v = CoordVector([1, 2, 3]);
        v.set_zero();
        assert_eq!(CoordVector([0, 0, 0]), v);
    }

    #[test]
    fn add() {
        assert_eq!(
            CoordVector([2, -1, 3]),
            CoordVector([1, 2, 3]) + CoordVector([1, -3, 0])
        );
    }

    #[test]
    fn add_assign() {
        let mut v = CoordVector([1, 2, 3]);
        v += CoordVector([1, -3, 0]);
        assert_eq!(CoordVector([2, -1, 3]), v);
    }

    #[test]
    fn sub() {
        assert_eq!(
            CoordVector([0, 5, 3]),
            CoordVector([1, 2, 3]) - CoordVector([1, -3, 0])
        );
    }

    #[test]
    fn sub_assign() {
        let mut v = CoordVector([1, 2, 3]);
        v -= CoordVector([1, -3, 0]);
        assert_eq!(CoordVector([0, 5, 3]), v);
    }

    #[test]
    fn neg() {
        assert_eq!(CoordVector([-1, 2, 0]), -CoordVector([1, -2, 0]));
    }

    #[test]
    fn mul() {
        assert_eq!(CoordVector([2, 4, 6]), CoordVector([1, 2, 3]) * 2);
    }

    #[test]
    fn mul_assign() {
        let mut v = CoordVector([1, 2, 3]);
        v *= 2;
        assert_eq!(CoordVector([2, 4, 6]), v);
    }

    #[test]
    fn div() {
        assert_eq!(CoordVector([1, 2, 3]), CoordVector([2, 4, 6]) / 2);
    }

    #[test]
    fn div_assign() {
        let mut v = CoordVector([2, 4, 6]);
        v /= 2;
        assert_eq!(CoordVector([1, 2, 3]), v);
    }

    #[test]
    fn mul_add() {
        assert_eq!(
            CoordVector([-1, -4, -4]),
            CoordVector([1, 2, 3]).mul_add(-2, CoordVector([1, 0, 2]))
        );
    }

    #[test]
    fn mul_add_assign() {
        let mut v = CoordVector([1, 2, 3]);
        v.mul_add_assign(-2, CoordVector([1, 0, 2]));
        assert_eq!(CoordVector([-1, -4, -4]), v);
    }
}
