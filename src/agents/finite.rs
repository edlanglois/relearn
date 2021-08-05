use super::{
    Actor, ActorMode, Agent, AgentBuilder, BatchUpdate, BuildAgentError, SetActorMode, Step,
};
use crate::envs::{EnvStructure, StoredEnvStructure};
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::{FiniteSpace, IndexSpace};

/// Wraps an index-space agent as an agent over finite spaces.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FiniteSpaceAgent<T, OS, AS> {
    pub agent: T,
    pub observation_space: OS,
    pub action_space: AS,
}

impl<T, OS, AS> FiniteSpaceAgent<T, OS, AS> {
    pub const fn new(agent: T, observation_space: OS, action_space: AS) -> Self {
        Self {
            agent,
            observation_space,
            action_space,
        }
    }
}

impl<T, OS, AS> Actor<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: Actor<usize, usize>, // Index-space actor
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        self.action_space
            .from_index(
                self.agent
                    .act(&self.observation_space.to_index(observation), new_episode),
            )
            .expect("Invalid action index")
    }
}

impl<T, OS, AS> Agent<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: Agent<usize, usize>, // Index-space agent
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn TimeSeriesLogger) {
        self.agent.update(
            indexed_step(&step, &self.observation_space, &self.action_space),
            logger,
        );
    }
}

impl<T, OS, AS> BatchUpdate<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: BatchUpdate<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn batch_update<I: IntoIterator<Item = Step<OS::Element, AS::Element>>>(
        &mut self,
        steps: I,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        let observation_space = &self.observation_space;
        let action_space = &self.action_space;
        self.agent.batch_update(
            steps
                .into_iter()
                .map(|s| indexed_step(&s, observation_space, action_space)),
            logger,
        );
        logger.end_event(Event::AgentOptPeriod);
    }
}

/// Convert a finite-space step into an index step.
fn indexed_step<OS, AS>(
    step: &Step<OS::Element, AS::Element>,
    observation_space: &OS,
    action_space: &AS,
) -> Step<usize, usize>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    Step {
        observation: observation_space.to_index(&step.observation),
        action: action_space.to_index(&step.action),
        reward: step.reward,
        next_observation: step
            .next_observation
            .as_ref()
            .map(|s| observation_space.to_index(s)),
        episode_done: step.episode_done,
    }
}

impl<T, OS, AS> SetActorMode for FiniteSpaceAgent<T, OS, AS>
where
    T: SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.agent.set_actor_mode(mode)
    }
}

impl<B, T, E> AgentBuilder<FiniteSpaceAgent<T, E::ObservationSpace, E::ActionSpace>, E> for B
where
    B: AgentBuilder<T, StoredEnvStructure<IndexSpace, IndexSpace>>,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: FiniteSpace,
    <E as EnvStructure>::ActionSpace: FiniteSpace,
{
    fn build_agent(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<FiniteSpaceAgent<T, E::ObservationSpace, E::ActionSpace>, BuildAgentError> {
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        let index_space_env = StoredEnvStructure {
            observation_space: (&observation_space).into(),
            action_space: (&action_space).into(),
            reward_range: env.reward_range(),
            discount_factor: env.discount_factor(),
        };
        Ok(FiniteSpaceAgent {
            agent: self.build_agent(&index_space_env, seed)?,
            observation_space,
            action_space,
        })
    }
}
