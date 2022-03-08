//! Feed-forward modules
mod activation;
mod linear;
mod mlp;

pub use activation::Activation;
pub use linear::{Linear, LinearConfig};
pub use mlp::{Mlp, MlpConfig};
