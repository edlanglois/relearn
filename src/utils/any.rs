//! Helpers for working with the [`Any`] trait
use std::any::Any;

/// Convert into [`Any`].
///
/// This is useful for converting an unsized trait object to `Any`.
/// # Important
/// Make sure that you call `as_ref` on the correct type.
/// For example, if `self: Box<T>` then `self.as_any()` gives you `<Box<T> as Any>`
/// but you probably want `self.as_ref().as_any()` for `<T as Any>`.
///
/// # Example
/// ```
/// use relearn::utils::any::AsAny;
/// use std::any::Any;
///
/// trait MyTrait: AsAny {
///     fn foo(&self) -> usize;
/// }
///
/// impl MyTrait for () {
///     fn foo(&self) -> usize {1}
/// }
///
/// let x: Box<dyn MyTrait> = Box::new(());
/// assert_eq!(x.foo(), 1);
///
/// let y: &dyn Any = x.as_ref().as_any();
/// let z: () = *(y.downcast_ref().unwrap());
/// assert_eq!(z.foo(), 1);
/// ```
pub trait AsAny {
    /// Convert into an [`Any`] trait reference.
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
