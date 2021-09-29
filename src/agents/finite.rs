use super::{
    Actor, ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError, SetActorMode, Step,
};
use crate::envs::EnvStructure;
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::FiniteSpace;

/// Build an agent for finite, indexed action and observation spaces.
///
/// This is a helper trait that automatically implements [`BuildAgent`]
/// with all finite-space environments.
pub trait BuildIndexAgent {
    type Agent;

    fn build_index_agent(
        &self,
        num_observations: usize,
        num_actions: usize,
        reward_range: (f64, f64),
        discount_factor: f64,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError>;
}

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
        logger.end_event(Event::AgentOptPeriod).unwrap();
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

impl<E, B> BuildAgent<E> for B
where
    B: BuildIndexAgent,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: FiniteSpace,
    <E as EnvStructure>::ActionSpace: FiniteSpace,
{
    type Agent =
        FiniteSpaceAgent<<Self as BuildIndexAgent>::Agent, E::ObservationSpace, E::ActionSpace>;

    fn build_agent(&self, env: &E, seed: u64) -> Result<Self::Agent, BuildAgentError> {
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        Ok(FiniteSpaceAgent {
            agent: self.build_index_agent(
                observation_space.size(),
                action_space.size(),
                env.reward_range(),
                env.discount_factor(),
                seed,
            )?,
            observation_space,
            action_space,
        })
    }
}
