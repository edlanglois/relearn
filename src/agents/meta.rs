//! Meta agents
use super::{
    buffers::NullBuffer, serial::SerialActorAgent, Actor, ActorMode, Agent, BatchUpdate,
    BufferCapacityBound, BuildAgent, BuildAgentError,
};
use crate::envs::{
    EnvStructure, InnerEnvStructure, MetaObservation, MetaObservationSpace, StoredEnvStructure,
    Successor,
};
use crate::logging::StatsLogger;
use crate::simulation::PartialStep;
use crate::spaces::{NonEmptySpace, Space};
use crate::Prng;
use std::fmt;
use std::sync::Arc;

/// Configuration for a [`ResettingMetaAgent`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResettingMetaAgentConfig<TC> {
    pub agent_config: TC,
}

impl<TC> ResettingMetaAgentConfig<TC> {
    pub const fn new(agent_config: TC) -> Self {
        Self { agent_config }
    }
}

impl<TC, OS, AS> BuildAgent<MetaObservationSpace<OS, AS>, AS> for ResettingMetaAgentConfig<TC>
where
    TC: BuildAgent<OS, AS> + Clone,
    OS: Space + Clone,
    AS: NonEmptySpace + Clone,
{
    type Agent = Arc<ResettingMetaAgent<TC, OS, AS>>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(Arc::new(ResettingMetaAgent::from_meta_env(
            self.agent_config.clone(),
            env,
        )))
    }
}

/// Lifts a regular agent to act on a meta environment (agent interface).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ResettingMetaAgent<TC, OS, AS> {
    inner_agent_config: TC,
    inner_env_structure: StoredEnvStructure<OS, AS>,
}

impl<TC, OS, AS> ResettingMetaAgent<TC, OS, AS> {
    pub fn new<E: ?Sized>(inner_agent_config: TC, inner_env_structure: &E) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
    {
        Self {
            inner_agent_config,
            inner_env_structure: StoredEnvStructure::from(inner_env_structure),
        }
    }
}

impl<TC, OS, AS> ResettingMetaAgent<TC, OS, AS>
where
    OS: Space,
    AS: Space,
{
    pub fn from_meta_env<E: ?Sized>(inner_agent_config: TC, env: &E) -> Self
    where
        E: EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
    {
        Self::new(inner_agent_config, &InnerEnvStructure::new(env))
    }
}

impl<TC, OS, AS> Agent<MetaObservation<OS::Element, AS::Element>, AS::Element>
    for Arc<ResettingMetaAgent<TC, OS, AS>>
where
    TC: BuildAgent<OS, AS>,
    OS: Space + Clone,
    AS: NonEmptySpace + Clone,
{
    type Actor = Self;

    fn actor(&self, _: ActorMode) -> Self {
        // Mode is ignored; this agent does not learn (at the meta level)
        self.clone() // Arc clone
    }
}

pub struct InnerEpisodeState<T, O, A>
where
    T: Agent<O, A>,
{
    inner_actor_agent: SerialActorAgent<T, O, A>,
    inner_actor_state: <T::Actor as Actor<O, A>>::EpisodeState,
    prev_observation: Option<O>,
}

impl<T, O, A> fmt::Debug for InnerEpisodeState<T, O, A>
where
    T: Agent<O, A> + fmt::Debug,
    T::Actor: fmt::Debug,
    T::HistoryBuffer: fmt::Debug,
    <T::Actor as Actor<O, A>>::EpisodeState: fmt::Debug,
    O: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerEpisodeState")
            .field("inner_actor_agent", &self.inner_actor_agent)
            .field("inner_actor_state", &self.inner_actor_state)
            .field("prev_observation", &self.prev_observation)
            .finish()
    }
}

impl<TC, OS, AS> Actor<MetaObservation<OS::Element, AS::Element>, AS::Element>
    for ResettingMetaAgent<TC, OS, AS>
where
    TC: BuildAgent<OS, AS>,
    OS: Space + Clone,
    AS: NonEmptySpace + Clone,
{
    type EpisodeState = InnerEpisodeState<TC::Agent, OS::Element, AS::Element>;

    fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        let inner_actor_agent = SerialActorAgent::new(
            self.inner_agent_config
                .build_agent(&self.inner_env_structure, rng)
                .expect("failed to build inner agent"),
        );
        InnerEpisodeState {
            inner_actor_state: inner_actor_agent.new_episode_state(rng),
            inner_actor_agent,
            prev_observation: None,
        }
    }

    fn act(
        &self,
        state: &mut Self::EpisodeState,
        obs: &MetaObservation<OS::Element, AS::Element>,
        rng: &mut Prng,
    ) -> AS::Element {
        // If the observation includes a previous step result then update the inner agent.
        if let Some(ref step_obs) = &obs.prev_step {
            let step_next = match (obs.inner_observation.as_ref(), obs.episode_done) {
                (Some(_), false) => Successor::Continue(()),
                (Some(o), true) => Successor::Interrupt(o.clone()),
                (None, true) => Successor::Terminate,
                (None, false) => panic!("missing successor observation for continuing episode"),
            };
            let step = PartialStep {
                observation: state.prev_observation.take().expect(
                    "meta observation follows a previous step but no previous observation stored",
                ),
                action: step_obs.action.clone(),
                reward: step_obs.reward,
                next: step_next,
            };
            state.inner_actor_agent.update(step, &mut ());
        }

        if obs.episode_done {
            // This observation marks the end of the inner episode. Reset the inner agent.
            state.inner_actor_state = state.inner_actor_agent.new_episode_state(rng);
            state.prev_observation = None;
            // The action will be ignored (since the inner episode is done); sample any action.
            self.inner_env_structure.action_space.some_element()
        } else {
            state.prev_observation = obs.inner_observation.as_ref().cloned();
            state.inner_actor_agent.act(
                &mut state.inner_actor_state,
                obs.inner_observation.as_ref().unwrap(),
                rng,
            )
        }
    }
}

/// No updates at the meta-level.
impl<TC, OS, AS> BatchUpdate<MetaObservation<OS::Element, AS::Element>, AS::Element>
    for Arc<ResettingMetaAgent<TC, OS, AS>>
where
    OS: Space,
    AS: Space,
{
    type HistoryBuffer = NullBuffer;

    fn batch_size_hint(&self) -> BufferCapacityBound {
        BufferCapacityBound::empty()
    }

    fn buffer(&self, _: BufferCapacityBound) -> Self::HistoryBuffer {
        NullBuffer
    }

    fn batch_update<'a, I>(&mut self, _buffers: I, _logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
    }
    fn batch_update_single(
        &mut self,
        _buffer: &mut Self::HistoryBuffer,
        _logger: &mut dyn StatsLogger,
    ) {
    }

    fn batch_update_slice(
        &mut self,
        _buffers: &mut [Self::HistoryBuffer],
        _logger: &mut dyn StatsLogger,
    ) {
    }
}

#[cfg(test)]
mod resetting_meta {
    use super::super::{ActorMode, UCB1AgentConfig};
    use super::*;
    use crate::envs::{Environment, MetaEnv, OneHotBandits};
    use crate::simulation::SimSeed;
    use rand::SeedableRng;

    #[test]
    fn ucb_one_hot_bandits() {
        let num_arms = 3;
        let num_episodes_per_trial = 20;
        let env = MetaEnv::new(OneHotBandits::new(num_arms), num_episodes_per_trial);
        let agent_config = ResettingMetaAgentConfig::new(UCB1AgentConfig::default());
        let agent = agent_config
            .build_agent(&env, &mut Prng::seed_from_u64(0))
            .unwrap();

        let mut total_episode_reward = 0.0;
        let mut current_episode_reward = 0.0;
        let mut num_episodes = 0;
        for step in env
            .run(agent.actor(ActorMode::Evaluation), SimSeed::Root(221), ())
            .take(1000)
        {
            current_episode_reward += step.reward;
            if step.next.continue_().is_none() {
                total_episode_reward += current_episode_reward;
                current_episode_reward = 0.0;
                num_episodes += 1;
            }
        }
        let mean_episode_reward = total_episode_reward / f64::from(num_episodes);
        assert!(mean_episode_reward > 0.7 * (num_episodes_per_trial - num_arms as u64) as f64);
    }
}
