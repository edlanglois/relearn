//! Environment builder traits
use super::{StructuredEnvDist, StructuredEnvironment, Wrapped};
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
    /// Environment feedback type.
    type Feedback: Clone + Send;
    /// Environment observation space type.
    type ObservationSpace: Space<Element = Self::Observation>;
    /// Environment action space type.
    type ActionSpace: Space<Element = Self::Action>;
    /// Environment feedback space type.
    type FeedbackSpace: Space<Element = Self::Feedback>;
    /// Type of environment to build
    type Environment: StructuredEnvironment<
        Observation = Self::Observation,
        Action = Self::Action,
        Feedback = Self::Feedback,
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
        FeedbackSpace = Self::FeedbackSpace,
    >;

    /// Build an environment instance.
    ///
    /// # Args
    /// * `rng` - Random number generator for randomness in the environment structure.
    fn build_env(&self, rng: &mut Prng) -> Result<Self::Environment, BuildEnvError>;

    /// Add a wrapper to the built environment
    #[inline]
    fn wrap<W>(self, wrapper: W) -> Wrapped<Self, W>
    where
        Wrapped<Self, W>: BuildEnv, // to help with debugging - fail immediately if invalid
        Self: Sized,
    {
        Wrapped {
            inner: self,
            wrapper,
        }
    }
}

impl<T: CloneBuild + StructuredEnvironment + ?Sized> BuildEnv for T {
    type Observation = T::Observation;
    type Action = T::Action;
    type Feedback = T::Feedback;
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;
    type FeedbackSpace = T::FeedbackSpace;
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
    type Feedback: Clone + Send;
    type ObservationSpace: Space<Element = Self::Observation>;
    type ActionSpace: Space<Element = Self::Action>;
    type FeedbackSpace: Space<Element = Self::Feedback>;
    type EnvDistribution: StructuredEnvDist<
        Observation = Self::Observation,
        Action = Self::Action,
        Feedback = Self::Feedback,
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
        FeedbackSpace = Self::FeedbackSpace,
    >;

    /// Build an environment distribution instance.
    fn build_env_dist(&self) -> Self::EnvDistribution;

    /// Wrap the environments in the built distribution.
    #[inline]
    fn wrap<W>(self, wrapper: W) -> Wrapped<Self, W>
    where
        Wrapped<Self, W>: BuildEnvDist, // to help with debugging - fail immediately if invalid
        Self: Sized,
    {
        Wrapped {
            inner: self,
            wrapper,
        }
    }
}

impl<T> BuildEnvDist for T
where
    T: StructuredEnvDist + CloneBuild,
{
    type Observation = <T::ObservationSpace as Space>::Element;
    type Action = <T::ActionSpace as Space>::Element;
    type Feedback = <T::FeedbackSpace as Space>::Element;
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;
    type FeedbackSpace = T::FeedbackSpace;
    type EnvDistribution = Self;

    #[inline]
    fn build_env_dist(&self) -> Self::EnvDistribution {
        self.clone()
    }
}
