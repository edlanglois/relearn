use super::Environment;
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
    fn build(&self, seed: u64) -> Result<E, BuildEnvError>;
}

/// Error building an environment
#[derive(Debug, Error)]
pub enum BuildEnvError {
    #[error(transparent)]
    BernoulliError(#[from] BernoulliError),
}

/// Non-stateful, cloneable environments can build themselves.
///
/// This allows using the environment as a configuration definition for itself,
/// avoiding the need for a duplicate structure to hold the same content,
/// and is useful for specifying wrapped environment configurations.
impl<E: Environment + Clone> EnvBuilder<E> for E {
    fn build(&self, _seed: u64) -> Result<E, BuildEnvError> {
        Ok(self.clone())
    }
}

impl From<Infallible> for BuildEnvError {
    fn from(_: Infallible) -> Self {
        unreachable!();
    }
}
