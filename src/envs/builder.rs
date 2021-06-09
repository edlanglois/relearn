use super::{DistWithState, EnvDistribution, Environment};
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

/// Clonable environment distributions can build themselves.
///
/// Allows using an environment distribution as its own configuration.
impl<E: EnvDistribution + Clone> EnvDistBuilder<Self> for E {
    fn build_env_dist(&self) -> Self {
        self.clone()
    }
}

/// Cloneable stateless environment can build a stateful version of themselves.
impl<E> EnvDistBuilder<DistWithState<Self>> for E
where
    E: EnvDistribution + Clone,
    <Self as EnvDistribution>::Environment: Environment,
{
    fn build_env_dist(&self) -> DistWithState<Self> {
        self.clone().into_stateful()
    }
}

impl From<Infallible> for BuildEnvError {
    fn from(_: Infallible) -> Self {
        unreachable!();
    }
}
