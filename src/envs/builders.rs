//! Environment builder traits
use super::{EnvDistribution, StructuredEnvironment};
use crate::spaces::Space;
use crate::Prng;
use std::convert::Infallible;
use thiserror::Error;

/// Marker indiciating that this object can build itself by cloning.
pub trait CloneBuild: Clone {}

/// Build a [`StructuredEnvironment`].
///
/// # Design Discussion
/// Environment is an associated trait rather than a generic parameter to facilitate
/// reproducibility: a given environment configuration can construct exactly one environment.
/// The user does not need to store the environment type.
pub trait BuildEnv {
    /// Environment observation type.
    type Observation: Clone + Send;
    /// Environment action type.
    type Action: Clone + Send;
    /// Environment observation space type.
    type ObservationSpace: Space<Element = Self::Observation>;
    /// Environment action space type.
    type ActionSpace: Space<Element = Self::Action>;
    /// Type of environment to build
    type Environment: StructuredEnvironment<
        Observation = Self::Observation,
        Action = Self::Action,
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
    >;

    /// Build an environment instance.
    ///
    /// # Args
    /// * `rng` - Random number generator for randomness in the environment structure.
    fn build_env(&self, rng: &mut Prng) -> Result<Self::Environment, BuildEnvError>;
}

impl<T: CloneBuild + StructuredEnvironment + ?Sized> BuildEnv for T {
    type Observation = T::Observation;
    type Action = T::Action;
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;
    type Environment = Self;

    fn build_env(&self, _: &mut Prng) -> Result<Self::Environment, BuildEnvError> {
        Ok(self.clone())
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

/// Build an [`EnvDistribution`].
pub trait BuildEnvDist {
    type Observation: Clone + Send;
    type Action: Clone + Send;
    type ObservationSpace: Space<Element = Self::Observation>;
    type ActionSpace: Space<Element = Self::Action>;
    type EnvDistribution: EnvDistribution<
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
    >;

    /// Build an environment distribution instance.
    fn build_env_dist(&self) -> Self::EnvDistribution;
}

impl<T> BuildEnvDist for T
where
    T: EnvDistribution + CloneBuild,
{
    type Observation = <T::ObservationSpace as Space>::Element;
    type Action = <T::ActionSpace as Space>::Element;
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;
    type EnvDistribution = Self;

    fn build_env_dist(&self) -> Self::EnvDistribution {
        self.clone()
    }
}
