mod step_limit;

pub use step_limit::{StepLimit, WithStepLimit};

use super::{BuildEnv, BuildEnvDist, BuildEnvError, EnvStructure, PomdpDistribution};
use rand::rngs::StdRng;

/// A basic wrapped object.
///
/// Consists of the inner object and the wrapper state.
///
/// May be used as an environment wrapper as `Wrapped<T, SomeWrapper>`
/// where `SomeWrapper: EnvWrapper + EnvStructureWrapper`
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
{
    type Pomdp = Wrapped<T::Pomdp, W>;

    fn sample_pomdp(&self, rng: &mut StdRng) -> Self::Pomdp {
        Wrapped::new(self.inner.sample_pomdp(rng), self.wrapper.clone())
    }
}

impl<B, E, W> BuildEnv<Wrapped<E, W>> for Wrapped<B, W>
where
    B: BuildEnv<E>,
    W: Copy,
{
    fn build_env(&self, seed: u64) -> Result<Wrapped<E, W>, BuildEnvError> {
        Ok(Wrapped {
            inner: self.inner.build_env(seed)?,
            wrapper: self.wrapper,
        })
    }
}

impl<B, D, W> BuildEnvDist<Wrapped<D, W>> for Wrapped<B, W>
where
    B: BuildEnvDist<D>,
    W: Copy,
{
    fn build_env_dist(&self) -> Wrapped<D, W> {
        Wrapped {
            inner: self.inner.build_env_dist(),
            wrapper: self.wrapper,
        }
    }
}
