//! Feed-forward modules
mod func;
mod linear;
mod mlp;

pub use func::{Activation, Func};
pub use linear::{Linear, LinearConfig};
pub use mlp::{Mlp, MlpConfig};
