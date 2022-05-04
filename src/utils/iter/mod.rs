//! Iterator utilities.
mod chain;
mod cmp;
mod rand;

pub use self::rand::RandSubsample;
pub use chain::SizedChain;
pub use cmp::{ArgMaxBy, PartialMax, PartialMaxError};
