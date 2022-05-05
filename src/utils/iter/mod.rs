//! Iterator utilities.
mod chain;
mod cmp;
mod rand;
mod split_chunks;

pub use self::rand::RandSubsample;
pub use chain::SizedChain;
pub use cmp::{ArgMaxBy, PartialMax, PartialMaxError};
pub use split_chunks::SplitChunksByLength;
