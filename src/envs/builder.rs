use rand::distributions::BernoulliError;
use std::convert::Infallible;
use thiserror::Error;

pub trait EnvBuilder<E> {
    /// Build an environment instance.
    ///
    /// # Args
    /// * `seed` - Seed for pseudo-randomness used by the environment.
    ///     Includes both randomization of the environment structure, and
    ///     random sampling of step outcomes within this structure.
    fn build_env(&self, seed: u64) -> Result<E, BuildEnvError>;
}

/// Error building an environment
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BuildEnvError {
    #[error(transparent)]
    BernoulliError(#[from] BernoulliError),
}

/// Builds an environment distribution.
pub trait EnvDistBuilder<E> {
    fn build_env_dist(&self) -> E;
}

impl From<Infallible> for BuildEnvError {
    fn from(_: Infallible) -> Self {
        unreachable!();
    }
}
