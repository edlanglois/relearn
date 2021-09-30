mod step_limit;

pub use step_limit::{StepLimit, WithStepLimit};

use super::{BuildEnvError, BuildPomdp, BuildPomdpDist, EnvStructure, Pomdp, PomdpDistribution};
use crate::spaces::Space;
use rand::rngs::StdRng;

// TODO: WrappedEnv that provides Env, EnvDistribution, BuildEnv, BuildEnvDist

/// A basic wrapped object.
///
/// Consists of the inner object and the wrapper state.
///
/// Cannot currently be used to wrap [`Environment`](super::Environment) or
/// [`EnvDistribution`](super::EnvDistribution).
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

impl<T, W> PomdpDistribution for Wrapped<T, W>
where
    T: PomdpDistribution,
    W: Clone,
    Wrapped<T::Pomdp, W>: Pomdp<
        // State = <T::Pomdp as Pomdp>::State,
        Observation = <T::Pomdp as Pomdp>::Observation,
        Action = <T::Pomdp as Pomdp>::Action,
    >,
{
    type Pomdp = Wrapped<T::Pomdp, W>;

    fn sample_pomdp(&self, rng: &mut StdRng) -> Self::Pomdp {
        Wrapped::new(self.inner.sample_pomdp(rng), self.wrapper.clone())
    }
}

impl<B, W> BuildPomdp for Wrapped<B, W>
where
    B: BuildPomdp,
    W: Clone,
    Wrapped<B::Pomdp, W>: Pomdp<Observation = B::Observation, Action = B::Action>,
{
    type State = <Self::Pomdp as Pomdp>::State;
    type Observation = <Self::Pomdp as Pomdp>::Observation;
    type Action = <Self::Pomdp as Pomdp>::Action;
    type ObservationSpace = <Self::Pomdp as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::Pomdp as EnvStructure>::ActionSpace;
    type Pomdp = Wrapped<B::Pomdp, W>;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        Ok(Wrapped::new(
            self.inner.build_pomdp()?,
            self.wrapper.clone(),
        ))
    }
}

impl<B, W> BuildPomdpDist for Wrapped<B, W>
where
    B: BuildPomdpDist,
    W: Clone,
    Wrapped<<B::PomdpDistribution as PomdpDistribution>::Pomdp, W>:
        Pomdp<Observation = B::Observation, Action = B::Action>,
{
    type Observation = <Self::ObservationSpace as Space>::Element;
    type Action = <Self::ActionSpace as Space>::Element;
    type ObservationSpace = <Self::PomdpDistribution as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::PomdpDistribution as EnvStructure>::ActionSpace;
    type PomdpDistribution = Wrapped<B::PomdpDistribution, W>;

    fn build_pomdp_dist(&self) -> Self::PomdpDistribution {
        Wrapped::new(self.inner.build_pomdp_dist(), self.wrapper.clone())
    }
}
