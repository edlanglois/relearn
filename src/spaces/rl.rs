use super::{ElementRefInto, FeatureSpace, ParameterizedDistributionSpace, SampleSpace, Space};
use crate::logging::Loggable;
use std::fmt::Debug;
use tch::Tensor;

/// A comprehensive space trait for use in reinforcement learning.
///
/// This includes all interfaces required by any agent, environment, or simulator
/// excluding interfaces that can only apply to some spaces, like [`FiniteSpace`].
///
/// [`FiniteSpace`]: super::FiniteSpace
pub trait RLSpace: Space + SampleSpace + ElementRefInto<Loggable> + Debug {}
impl<T: Space + SampleSpace + ElementRefInto<Loggable> + Debug> RLSpace for T {}

/// Comprehensive observation space for use in reinforcement learning
pub trait RLObservationSpace: RLSpace + FeatureSpace<Tensor> {}
impl<T: RLSpace + FeatureSpace<Tensor>> RLObservationSpace for T {}

/// Comprehensive action space for use in reinforcement learning
pub trait RLActionSpace: RLSpace + ParameterizedDistributionSpace<Tensor> {}
impl<T: RLSpace + ParameterizedDistributionSpace<Tensor>> RLActionSpace for T {}
