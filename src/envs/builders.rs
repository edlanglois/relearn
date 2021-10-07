//! Environment builder traits
use super::{
    EnvDistribution, EnvStructure, Environment, IntoEnv, Pomdp, PomdpDistribution, PomdpEnv,
};
use crate::spaces::Space;
use std::convert::Infallible;
use thiserror::Error;

/// Marker indiciating that this object can build itself by cloning.
pub trait CloneBuild: Clone {}

/// Build a [`Pomdp`].
pub trait BuildPomdp {
    type State;
    type Observation;
    type Action;
    type ObservationSpace: Space<Element = Self::Observation>;
    type ActionSpace: Space<Element = Self::Action>;
    type Pomdp: Pomdp<State = Self::State, Observation = Self::Observation, Action = Self::Action>
        + EnvStructure<ObservationSpace = Self::ObservationSpace, ActionSpace = Self::ActionSpace>;

    /// Build a [`Pomdp`] instance.
    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError>;
}

impl<E> BuildPomdp for E
where
    E: Pomdp + EnvStructure + CloneBuild,
    E::ObservationSpace: Space<Element = E::Observation>,
    E::ActionSpace: Space<Element = E::Action>,
{
    type State = E::State;
    type Observation = E::Observation;
    type Action = E::Action;
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;
    type Pomdp = Self;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        Ok(self.clone())
    }
}

/// Build an [`Environment`](super::Environment).
///
/// Environment is an associated trait rather than a generic parameter to facilitate
/// reproducibility: a given environment configuration can construct exactly one environment.
/// The user does not need to store the environment type.
pub trait BuildEnv {
    /// Environment observation type.
    type Observation;
    /// Environment action type.
    type Action;
    /// Environment observation space type.
    type ObservationSpace: Space<Element = Self::Observation>;
    /// Environment action space type.
    type ActionSpace: Space<Element = Self::Action>;
    /// Type of environment to build
    type Environment: Environment<Observation = Self::Observation, Action = Self::Action>
        + EnvStructure<ObservationSpace = Self::ObservationSpace, ActionSpace = Self::ActionSpace>;

    /// Build an environment instance.
    ///
    /// # Args
    /// * `seed` - Seed for pseudo-randomness in the environment state and transition dynamics.
    ///            The environment structure itself should not be random; seed any structural
    ///            randomness from the environment configuration.
    fn build_env(&self, seed: u64) -> Result<Self::Environment, BuildEnvError>;
}

impl<T: BuildPomdp> BuildEnv for T {
    type Observation = T::Observation;
    type Action = T::Action;
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;
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
    type Observation;
    type Action;
    type ObservationSpace: Space<Element = Self::Observation>;
    type ActionSpace: Space<Element = Self::Action>;
    type PomdpDistribution: PomdpDistribution<
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
    >;

    /// Build a POMDP distribution instance.
    fn build_pomdp_dist(&self) -> Self::PomdpDistribution;
}

impl<T> BuildPomdpDist for T
where
    T: PomdpDistribution + CloneBuild,
{
    type Observation = <T::ObservationSpace as Space>::Element;
    type Action = <T::ActionSpace as Space>::Element;
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;
    type PomdpDistribution = Self;

    fn build_pomdp_dist(&self) -> Self::PomdpDistribution {
        self.clone()
    }
}

/// Build an [`EnvDistribution`].
pub trait BuildEnvDist {
    type Observation;
    type Action;
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
