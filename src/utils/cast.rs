//! Casting

/// Lossy conversion to another type.
pub trait CastInto<T> {
    fn cast_into(self) -> T;
}

impl<T> CastInto<T> for T {
    fn cast_into(self) -> T {
        self
    }
}

impl CastInto<f64> for f32 {
    fn cast_into(self) -> f64 {
        self.into()
    }
}

impl CastInto<f32> for f64 {
    fn cast_into(self) -> f32 {
        self as f32
    }
}
