//! Feed-forward modules
mod func;
mod mlp;

pub use func::{Activation, Func};
pub use mlp::{Mlp, MlpConfig};
