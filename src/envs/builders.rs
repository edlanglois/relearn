//! Environment builder traits
use super::{EnvDistribution, IntoEnv, Pomdp, PomdpDistribution, PomdpEnv};
use std::convert::Infallible;
use thiserror::Error;

/// Marker indiciating that this object can build itself by cloning.
pub trait CloneBuild: Clone {}

/// Build a [`Pomdp`].
pub trait BuildPomdp {
    type Pomdp;

    /// Build a [`Pomdp`] instance.
    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError>;
}

impl<E: Pomdp + CloneBuild> BuildPomdp for E {
    type Pomdp = Self;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        Ok(self.clone())
    }
}

/// Build an [`Environment`].
///
/// Environment is an associated trait rather than a generic parameter to facilitate
/// reproducibility: a given environment configuration can construct exactly one environment.
/// The user does not need to store the environment type.
pub trait BuildEnv {
    type Environment;

    /// Build an environment instance.
    ///
    /// # Args
    /// * `seed` - Seed for pseudo-randomness in the environment state and transition dynamics.
    ///            The environment structure itself should not be random; seed any structural
    ///            randomness from the environment configuration.
    fn build_env(&self, seed: u64) -> Result<Self::Environment, BuildEnvError>;
}

impl<T> BuildEnv for T
where
    T: BuildPomdp,
    <T as BuildPomdp>::Pomdp: Pomdp,
{
    type Environment = PomdpEnv<T::Pomdp>;

    fn build_env(&self, seed: u64) -> Result<Self::Environment, BuildEnvError> {
        Ok(self.build_pomdp()?.into_env(seed))
    }
}

/// Error building an environment
#[derive(Debug, Error)]
pub enum BuildEnvError {
    #[error(transparent)]
    Boxed(#[from] Box<dyn std::error::Error>),
}

impl From<Infallible> for BuildEnvError {
    fn from(_: Infallible) -> Self {
        unreachable!();
    }
}

/// Build a [`PomdpDistribution`].
pub trait BuildPomdpDist {
    type PomdpDistribution;

    /// Build a POMDP distribution instance.
    fn build_pomdp_dist(&self) -> Self::PomdpDistribution;
}

impl<T> BuildPomdpDist for T
where
    T: PomdpDistribution + CloneBuild,
{
    type PomdpDistribution = Self;

    fn build_pomdp_dist(&self) -> Self::PomdpDistribution {
        self.clone()
    }
}

/// Build an [`EnvDistribution`].
pub trait BuildEnvDist {
    type EnvDistribution;

    /// Build an environment distribution instance.
    fn build_env_dist(&self) -> Self::EnvDistribution;
}

impl<T> BuildEnvDist for T
where
    T: EnvDistribution + CloneBuild,
{
    type EnvDistribution = Self;

    fn build_env_dist(&self) -> Self::EnvDistribution {
        self.clone()
    }
}
