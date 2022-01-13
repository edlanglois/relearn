mod step_limit;

pub use step_limit::{StepLimit, WithStepLimit};

use super::{BuildEnv, BuildEnvDist, BuildEnvError, EnvDistribution, EnvStructure, Environment};
use crate::spaces::Space;
use crate::Prng;

/// A basic wrapped object.
///
/// Consists of the inner object and the wrapper state.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
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

// TODO: Allow wrappers that change the structure types
impl<T: EnvStructure, W> EnvStructure for Wrapped<T, W> {
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space()
    }

    fn reward_range(&self) -> (f64, f64) {
        self.inner.reward_range()
    }

    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<EC, W> BuildEnv for Wrapped<EC, W>
where
    EC: BuildEnv,
    W: Clone,
    Wrapped<EC::Environment, W>: Environment<Observation = EC::Observation, Action = EC::Action>,
{
    type Observation = <Self::Environment as Environment>::Observation;
    type Action = <Self::Environment as Environment>::Action;
    type ObservationSpace = <Self::Environment as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::Environment as EnvStructure>::ActionSpace;
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
    Wrapped<ED::Environment, W>: Environment<
        Observation = <ED::Environment as Environment>::Observation,
        Action = <ED::Environment as Environment>::Action,
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
    >,
{
    type Observation = <Self::ObservationSpace as Space>::Element;
    type Action = <Self::ActionSpace as Space>::Element;
    type ObservationSpace = <Self::EnvDistribution as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::EnvDistribution as EnvStructure>::ActionSpace;
    type EnvDistribution = Wrapped<EDC::EnvDistribution, W>;

    fn build_env_dist(&self) -> Self::EnvDistribution {
        Wrapped {
            inner: self.inner.build_env_dist(),
            wrapper: self.wrapper.clone(),
        }
    }
}
