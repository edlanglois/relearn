mod step_limit;

pub use step_limit::{
    LatentStepLimit, VisibleStepLimit, WithLatentStepLimit, WithVisibleStepLimit,
};

use super::{
    BuildEnv, BuildEnvDist, BuildEnvError, EnvDistribution, EnvStructure, Environment,
    StructuredEnvDist, StructuredEnvironment,
};
use crate::Prng;
use serde::{Deserialize, Serialize};

/// Trait providing a `wrap` method for all sized types.
pub trait Wrap: Sized {
    /// Wrap in the given wrapper.
    #[inline]
    fn wrap<W>(self, wrapper: W) -> Wrapped<Self, W> {
        Wrapped {
            inner: self,
            wrapper,
        }
    }
}

impl<T> Wrap for T {}

/// A basic wrapped object.
///
/// Consists of the inner object and the wrapper state.
///
/// # Implementation
/// To implement a wrapper type, define `struct MyWrapper` and implement
/// `impl<T: Environment> Environment for Wrapped<T, MyWrapper>` and
/// `impl<T: EnvStructure> EnvStructure for Wrapped<T, MyWrapper>`.
///
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Wrapped<T, W> {
    /// Wrapped object
    pub inner: T,
    /// The wrapper
    pub wrapper: W,
}

impl<T, W> Wrapped<T, W> {
    pub const fn new(inner: T, wrapper: W) -> Self {
        Self { inner, wrapper }
    }
}

/// Marker trait for a wrapper that does not modify the environment structure.
pub trait StructurePreservingWrapper {}

impl<E, W> EnvStructure for Wrapped<E, W>
where
    E: EnvStructure,
    W: StructurePreservingWrapper,
{
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;
    type FeedbackSpace = E::FeedbackSpace;

    #[inline]
    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space()
    }
    #[inline]
    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space()
    }
    #[inline]
    fn feedback_space(&self) -> Self::FeedbackSpace {
        self.inner.feedback_space()
    }
    #[inline]
    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<EC, W> BuildEnv for Wrapped<EC, W>
where
    EC: BuildEnv,
    W: Clone,
    Wrapped<EC::Environment, W>: StructuredEnvironment,
{
    type Observation = <Self::Environment as Environment>::Observation;
    type Action = <Self::Environment as Environment>::Action;
    type Feedback = <Self::Environment as Environment>::Feedback;
    type ObservationSpace = <Self::Environment as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::Environment as EnvStructure>::ActionSpace;
    type FeedbackSpace = <Self::Environment as EnvStructure>::FeedbackSpace;
    type Environment = Wrapped<EC::Environment, W>;

    #[inline]
    fn build_env(&self, rng: &mut Prng) -> Result<Self::Environment, BuildEnvError> {
        Ok(Wrapped {
            inner: self.inner.build_env(rng)?,
            wrapper: self.wrapper.clone(),
        })
    }
}

impl<ED, W> EnvDistribution for Wrapped<ED, W>
where
    ED: EnvDistribution,
    W: Clone,
    Wrapped<ED::Environment, W>: Environment,
{
    type State = <Self::Environment as Environment>::State;
    type Observation = <Self::Environment as Environment>::Observation;
    type Action = <Self::Environment as Environment>::Action;
    type Feedback = <Self::Environment as Environment>::Feedback;
    type Environment = Wrapped<ED::Environment, W>;

    #[inline]
    fn sample_environment(&self, rng: &mut Prng) -> Self::Environment {
        Wrapped {
            inner: self.inner.sample_environment(rng),
            wrapper: self.wrapper.clone(),
        }
    }
}

impl<EDC, W> BuildEnvDist for Wrapped<EDC, W>
where
    EDC: BuildEnvDist,
    W: Clone,
    Wrapped<EDC::EnvDistribution, W>: StructuredEnvDist,
{
    type Observation = <Self::EnvDistribution as EnvDistribution>::Observation;
    type Action = <Self::EnvDistribution as EnvDistribution>::Action;
    type Feedback = <Self::EnvDistribution as EnvDistribution>::Feedback;
    type ObservationSpace = <Self::EnvDistribution as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::EnvDistribution as EnvStructure>::ActionSpace;
    type FeedbackSpace = <Self::EnvDistribution as EnvStructure>::FeedbackSpace;
    type EnvDistribution = Wrapped<EDC::EnvDistribution, W>;

    #[inline]
    fn build_env_dist(&self) -> Self::EnvDistribution {
        Wrapped {
            inner: self.inner.build_env_dist(),
            wrapper: self.wrapper.clone(),
        }
    }
}
