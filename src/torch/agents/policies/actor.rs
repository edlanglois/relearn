use crate::agents::Actor;
use crate::spaces::{FeatureSpace, NonEmptyFeatures, ParameterizedDistributionSpace};
use crate::torch::modules::SeqIterative;
use crate::Prng;
use serde::{Deserialize, Serialize};
use tch::Tensor;

/// An [`Actor`] that samples actions according to a policy module.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PolicyActor<OS, AS, P> {
    observation_space: NonEmptyFeatures<OS>,
    action_space: AS,
    policy_module: P,
}

impl<OS, AS, P> PolicyActor<OS, AS, P> {
    pub const fn new(
        observation_space: NonEmptyFeatures<OS>,
        action_space: AS,
        policy_module: P,
    ) -> Self {
        Self {
            observation_space,
            action_space,
            policy_module,
        }
    }
}

impl<OS, AS, P> Actor<OS::Element, AS::Element> for PolicyActor<OS, AS, P>
where
    OS: FeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: SeqIterative,
{
    type EpisodeState = P::State;

    fn initial_state(&self, _: &mut Prng) -> Self::EpisodeState {
        self.policy_module.initial_state()
    }

    fn act(
        &self,
        episode_state: &mut Self::EpisodeState,
        observation: &OS::Element,
        _: &mut Prng,
    ) -> AS::Element {
        let _no_grad = tch::no_grad_guard();
        let observation_features: Tensor = self.observation_space.features(observation);
        let action_distribution_params = self
            .policy_module
            .step(episode_state, &observation_features);
        self.action_space
            .sample_element(&action_distribution_params)
    }
}
