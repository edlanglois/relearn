use super::{ElementRefInto, FeatureSpace, ParameterizedSampleSpace, SampleSpace, Space};
use crate::logging::Loggable;
use std::fmt::Debug;
use tch::Tensor;

/// A comprehensive space type for use in reinforcement learning.
///
/// This includes all interfaces required by any agent, environment, or simulator
/// excluding interfaces that can only apply to some spaces, like FiniteSpace.
pub trait RLSpace:
    Space
    + SampleSpace
    + FeatureSpace<Tensor>
    + ParameterizedSampleSpace<Tensor>
    + ElementRefInto<Loggable>
    + Debug
{
}
impl<
        T: Space
            + SampleSpace
            + FeatureSpace<Tensor>
            + ParameterizedSampleSpace<Tensor>
            + ElementRefInto<Loggable>
            + Debug,
    > RLSpace for T
{
}
