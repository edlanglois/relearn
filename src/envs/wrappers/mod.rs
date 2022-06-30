mod step_limit;

pub use step_limit::{
    LatentStepLimit, VisibleStepLimit, WithLatentStepLimit, WithVisibleStepLimit,
};

use super::{
    BuildEnv, BuildEnvDist, BuildEnvError, EnvDistribution, EnvStructure, Environment,
    StructuredEnvironment,
};
use crate::spaces::Space;
use crate::Prng;
use serde::{Deserialize, Serialize};

/// A basic wrapped object.
///
/// Consists of the inner object and the wrapper state.
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
    Self: EnvStructure,
    Wrapped<ED::Environment, W>: StructuredEnvironment<
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
        FeedbackSpace = Self::FeedbackSpace,
    >,
{
    type Environment = Wrapped<ED::Environment, W>;

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
    Wrapped<EDC::EnvDistribution, W>: EnvDistribution<
        ObservationSpace = <EDC::EnvDistribution as EnvStructure>::ObservationSpace,
        ActionSpace = <EDC::EnvDistribution as EnvStructure>::ActionSpace,
        FeedbackSpace = <EDC::EnvDistribution as EnvStructure>::FeedbackSpace,
    >,
{
    type Observation = <Self::ObservationSpace as Space>::Element;
    type Action = <Self::ActionSpace as Space>::Element;
    type Feedback = <Self::FeedbackSpace as Space>::Element;
    type ObservationSpace = <Self::EnvDistribution as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::EnvDistribution as EnvStructure>::ActionSpace;
    type FeedbackSpace = <Self::EnvDistribution as EnvStructure>::FeedbackSpace;
    type EnvDistribution = Wrapped<EDC::EnvDistribution, W>;

    fn build_env_dist(&self) -> Self::EnvDistribution {
        Wrapped {
            inner: self.inner.build_env_dist(),
            wrapper: self.wrapper.clone(),
        }
    }
}
