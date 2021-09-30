//! Meta agents
use super::{Actor, Agent, BuildAgent, BuildAgentError, SetActorMode, Step};
use crate::envs::{EnvStructure, InnerEnvStructure, MetaObservationSpace, StoredEnvStructure};
use crate::logging::TimeSeriesLogger;
use crate::spaces::{SampleSpace, Space};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Configuration for a [`ResettingMetaAgent`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResettingMetaAgentConfig<T> {
    pub agent_config: T,
}

impl<T> ResettingMetaAgentConfig<T> {
    pub const fn new(agent_config: T) -> Self {
        Self { agent_config }
    }
}

impl<T, E, OS, AS> BuildAgent<E> for ResettingMetaAgentConfig<T>
where
    T: BuildAgent<StoredEnvStructure<OS, AS>> + Clone,
    E: EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
    OS: Space + Clone,
    <OS as Space>::Element: Clone,
    AS: SampleSpace + Clone,
    <AS as Space>::Element: Clone,
{
    type Agent = ResettingMetaAgent<T, OS, AS>;

    fn build_agent(&self, env: &E, seed: u64) -> Result<Self::Agent, BuildAgentError> {
        let inner_env_structure = StoredEnvStructure::from(&InnerEnvStructure::<E, &E>::new(env));
        ResettingMetaAgent::new(self.agent_config.clone(), inner_env_structure, seed)
    }
}

/// Lifts a regular agent to act on a meta environment by resetting between each trial.
pub struct ResettingMetaAgent<AC, OS, AS>
where
    AC: BuildAgent<StoredEnvStructure<OS, AS>>,
    OS: Space + Clone,
    AS: Space + Clone,
{
    inner_agent_config: AC,
    inner_env_structure: StoredEnvStructure<OS, AS>,
    rng: StdRng,
    agent: AC::Agent,
    prev_observation: Option<OS::Element>,
    prev_episode_done: bool,
}

impl<AC, OS, AS> ResettingMetaAgent<AC, OS, AS>
where
    AC: BuildAgent<StoredEnvStructure<OS, AS>>,
    OS: Space + Clone,
    AS: Space + Clone,
{
    /// Intialize a new resetting meta agent.
    ///
    /// # Args
    /// * `inner_agent_config` - A builder for the inner agent.
    /// * `inner_env_structure` - The inner environment structure.
    /// * `seed` - Seeds the internal pseudo random state.
    pub fn new(
        inner_agent_config: AC,
        inner_env_structure: StoredEnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Self, BuildAgentError> {
        let mut rng = StdRng::seed_from_u64(seed);
        let agent = inner_agent_config.build_agent(&inner_env_structure, rng.gen())?;
        Ok(Self {
            inner_agent_config,
            inner_env_structure,
            rng,
            agent,
            prev_observation: None,
            prev_episode_done: true,
        })
    }
}

impl<AC, OS, AS> Actor<<MetaObservationSpace<OS, AS> as Space>::Element, AS::Element>
    for ResettingMetaAgent<AC, OS, AS>
where
    AC: BuildAgent<StoredEnvStructure<OS, AS>>,
    OS: Space + Clone,
    <OS as Space>::Element: Clone,
    AS: SampleSpace + Clone,
    <AS as Space>::Element: Clone,
{
    fn act(
        &mut self,
        observation: &<MetaObservationSpace<OS, AS> as Space>::Element,
        new_episode: bool,
    ) -> AS::Element {
        let (inner_observation, step_info, episode_done) = observation;

        if new_episode {
            // Reset the agent
            self.agent = self
                .inner_agent_config
                .build_agent(&self.inner_env_structure, self.rng.gen())
                .expect("Failed to build inner agent");
            self.prev_observation = None;
            self.prev_episode_done = true;
        } else if let Some((action, reward)) = step_info {
            // Update the agent based on the most recent step result
            // Only relevant if the agent has not been reset.
            let step = Step {
                observation: self.prev_observation.take().expect(
                    "Meta observation follows a previous step but no previous observation stored",
                ),
                action: action.clone(),
                reward: *reward,
                next_observation: inner_observation.as_ref().cloned(),
                episode_done: *episode_done,
            };
            self.agent.update(step, &mut ());
        }

        let action = if let Some(inner_observation) = inner_observation {
            self.agent.act(inner_observation, self.prev_episode_done)
        } else {
            // If there is no inner observation then the current state is terminal
            // and the inner episode is done so whatever this action is, it will be ignored.
            assert!(
                *episode_done,
                "Expecting episode_done if inner_observation is None"
            );
            // TODO: Replace with a non-random some_element() method
            self.inner_env_structure.action_space.sample(&mut self.rng)
        };

        self.prev_observation = inner_observation.as_ref().cloned();
        self.prev_episode_done = *episode_done;

        action
    }
}

impl<AC, OS, AS> Agent<<MetaObservationSpace<OS, AS> as Space>::Element, AS::Element>
    for ResettingMetaAgent<AC, OS, AS>
where
    AC: BuildAgent<StoredEnvStructure<OS, AS>>,
    OS: Space + Clone,
    <OS as Space>::Element: Clone,
    AS: SampleSpace + Clone,
    <AS as Space>::Element: Clone,
{
    fn update(
        &mut self,
        _step: Step<<MetaObservationSpace<OS, AS> as Space>::Element, AS::Element>,
        _logger: &mut dyn TimeSeriesLogger,
    ) {
        // Does not learn on a meta level
    }
}

/// Never learns on a meta level. Always acts like "Release" mode.
impl<AC, OS, AS> SetActorMode for ResettingMetaAgent<AC, OS, AS>
where
    AC: BuildAgent<StoredEnvStructure<OS, AS>>,
    OS: Space + Clone,
    AS: Space + Clone,
{
}

#[cfg(test)]
mod resetting_meta {
    use super::super::{ActorMode, UCB1AgentConfig};
    use super::*;
    use crate::envs::{MetaPomdp, OneHotBandits, PomdpEnv};
    use crate::simulation;
    use crate::simulation::hooks::{RewardStatistics, StepLimit};

    #[test]
    fn ucb_one_hot_bandits() {
        let config = ResettingMetaAgentConfig::new(UCB1AgentConfig::default());
        let num_arms = 3;
        let num_episodes_per_trial = 20;
        let mut env = PomdpEnv::new(
            MetaPomdp::new(OneHotBandits::new(num_arms), num_episodes_per_trial),
            0,
        );
        let mut agent = config.build_agent(&env, 0).unwrap();

        let mut hooks = (RewardStatistics::new(), StepLimit::new(1000));
        agent.set_actor_mode(ActorMode::Release);
        simulation::run_actor(&mut env, &mut agent, &mut hooks, &mut ());

        assert!(hooks.0.mean_episode_reward() > 0.7 * (num_episodes_per_trial - num_arms) as f64);
    }
}
