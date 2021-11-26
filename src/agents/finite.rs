use super::buffers::HistoryBuffer;
use super::{
    Actor, ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError, BuildBatchUpdateActor,
    OffPolicyAgent, SetActorMode, Step, SyncParams, SyncParamsError,
};
use crate::envs::EnvStructure;
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::{FiniteSpace, IndexSpace, Space};

/// Build an agent for finite, indexed action and observation spaces.
///
/// This is a helper trait that automatically implements [`BuildAgent`]
/// with all finite-space environments.
pub trait BuildIndexAgent {
    type Agent: Agent<<IndexSpace as Space>::Element, <IndexSpace as Space>::Element> + SetActorMode;

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

/// Perform a batch update from a in iterator of steps by value.
///
/// Used for finite spaces to help implement [`BatchUpdate`] for [`FiniteSpaceAgent`].
/// The [`HistoryBuffer`] interface only returns iterators of references which makes it
/// inconvenient to wrap a `HistoryBuffer<O, A>` as a `HistoryBuffer<usize, usize>`.
///
/// One approach could be to allow [`BatchUpdateAgent`] to use a history buffer with a different
/// type from the external interface, along with a function `Fn(Step<O, A>) -> Step<O2, A2>`
/// applied before pushing to the buffer.
///
/// Alternatively, since current finite space batch updates only perform one pass over the steps,
/// a simpler solution is to use this alternate interface for batch updates. Since it takes steps
/// by value, it is possibly to map the steps without having to allocate a new vector.
pub trait BatchUpdateFromSteps<O, A> {
    fn batch_update_from_steps<I: IntoIterator<Item = Step<O, A>>>(
        &mut self,
        steps: I,
        logger: &mut dyn TimeSeriesLogger,
    );
}

impl<T, O, A> BatchUpdateFromSteps<O, A> for T
where
    T: OffPolicyAgent + Agent<O, A>,
{
    fn batch_update_from_steps<I: IntoIterator<Item = Step<O, A>>>(
        &mut self,
        steps: I,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        for step in steps {
            self.update(step, logger);
        }
        logger.end_event(Event::AgentOptPeriod).unwrap()
    }
}

impl<T, OS, AS> BatchUpdate<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: BatchUpdateFromSteps<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn batch_update(
        &mut self,
        history: &mut dyn HistoryBuffer<OS::Element, AS::Element>,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        self.agent.batch_update_from_steps(
            history
                .steps()
                .map(|step| indexed_step(step, &self.observation_space, &self.action_space)),
            logger,
        )
    }
}

impl<T, OS, AS> SyncParams for FiniteSpaceAgent<T, OS, AS>
where
    T: SyncParams,
{
    fn sync_params(&mut self, target: &Self) -> Result<(), SyncParamsError> {
        self.agent.sync_params(&target.agent)
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

impl<B, OS, AS> BuildAgent<OS, AS> for B
where
    B: BuildIndexAgent,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    type Agent = FiniteSpaceAgent<B::Agent, OS, AS>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
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

impl<B, OS, AS> BuildBatchUpdateActor<OS, AS> for B
where
    // NOTE: This is slightly over-restrictive. Don't need BuildIndexAgent::Agent: Agent
    B: BuildIndexAgent,
    B::Agent: BatchUpdateFromSteps<usize, usize>,
    OS: FiniteSpace,
    OS::Element: 'static,
    AS: FiniteSpace,
    AS::Element: 'static,
{
    type BatchUpdateActor = <Self as BuildAgent<OS, AS>>::Agent;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        self.build_agent(env, seed)
    }
}
