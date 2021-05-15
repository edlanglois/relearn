use super::{
    ElementRefInto, FeatureSpace, ParameterizedDistributionSpace, ReprSpace, SampleSpace, Space,
};
use crate::logging::Loggable;
use std::fmt::Debug;
use tch::Tensor;

/// A comprehensive space trait for use in reinforcement learning.
///
/// This includes all interfaces required by any agent, environment, or simulator
/// excluding interfaces that can only apply to some spaces, like [`FiniteSpace`].
///
/// [`FiniteSpace`]: super::FiniteSpace
pub trait RLSpace:
    Space
    + SampleSpace
    + FeatureSpace<Tensor> // Observations only
    + ReprSpace<Tensor> // Actions only
    + ParameterizedDistributionSpace<Tensor> // Actions only
    + ElementRefInto<Loggable>
    + Debug
{
}
impl<
        T: Space
            + SampleSpace
            + FeatureSpace<Tensor>
            + ReprSpace<Tensor>
            + ParameterizedDistributionSpace<Tensor>
            + ElementRefInto<Loggable>
            + Debug,
    > RLSpace for T
{
}
