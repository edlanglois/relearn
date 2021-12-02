use super::{
    Actor, ActorMode, AsyncUpdate, BatchUpdate, BuildAgent, BuildAgentError, MakeActor, PureActor,
    SetActorMode, SynchronousUpdate, WriteHistoryBuffer,
};
use crate::envs::{EnvStructure, Successor};
use crate::logging::TimeSeriesLogger;
use crate::simulation::{PartialStep, Step, TransientStep};
use crate::spaces::{FiniteSpace, IndexSpace, Space};

/// Build an agent for finite, indexed action and observation spaces.
///
/// This is a helper trait that automatically implements [`BuildAgent`]
/// with all finite-space environments.
pub trait BuildIndexAgent {
    type Agent: SynchronousUpdate<<IndexSpace as Space>::Element, <IndexSpace as Space>::Element>
        + SetActorMode;

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

impl<T, OS, AS> PureActor<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: PureActor<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    type State = T::State;

    fn initial_state(&self, seed: u64) -> Self::State {
        self.agent.initial_state(seed)
    }

    fn reset_state(&self, state: &mut Self::State) {
        self.agent.reset_state(state)
    }

    fn act(&self, state: &mut Self::State, observation: &OS::Element) -> AS::Element {
        self.action_space
            .from_index(
                self.agent
                    .act(state, &self.observation_space.to_index(observation)),
            )
            .expect("invalid action index")
    }
}

impl<T, OS, AS> Actor<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: Actor<usize, usize>, // Index-space actor
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn act(&mut self, observation: &OS::Element) -> AS::Element {
        self.action_space
            .from_index(
                self.agent
                    .act(&self.observation_space.to_index(observation)),
            )
            .expect("invalid action index")
    }

    fn reset(&mut self) {
        self.agent.reset()
    }
}

impl<T, OS, AS> SynchronousUpdate<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: SynchronousUpdate<usize, usize>, // Index-space agent
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn update(
        &mut self,
        step: TransientStep<OS::Element, AS::Element>,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        let step = indexed_step(&step, &self.observation_space, &self.action_space);
        let transient_step = TransientStep {
            observation: step.observation,
            action: step.action,
            reward: step.reward,
            next: match step.next.as_ref() {
                Successor::Continue(o) => Successor::Continue(o),
                Successor::Terminate => Successor::Terminate,
                Successor::Interrupt(o) => Successor::Interrupt(*o),
            },
        };
        self.agent.update(transient_step, logger);
    }
}

impl<T: AsyncUpdate, OS, AS> AsyncUpdate for FiniteSpaceAgent<T, OS, AS> {}

// Wrap an index-space buffer to accept finite-space elements.
pub struct FiniteSpaceBuffer<B, OS, AS> {
    buffer: B,
    observation_space: OS,
    action_space: AS,
}

impl<B, OS, AS> WriteHistoryBuffer<OS::Element, AS::Element> for FiniteSpaceBuffer<B, OS, AS>
where
    B: WriteHistoryBuffer<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn push(&mut self, step: PartialStep<OS::Element, AS::Element>) -> bool {
        self.buffer.push(indexed_partial_step(
            &step,
            &self.observation_space,
            &self.action_space,
        ))
    }

    fn extend<I>(&mut self, steps: I) -> bool
    where
        I: IntoIterator<Item = PartialStep<OS::Element, AS::Element>>,
    {
        self.buffer.extend(
            steps.into_iter().map(|step| {
                indexed_partial_step(&step, &self.observation_space, &self.action_space)
            }),
        )
    }

    fn clear(&mut self) {
        self.buffer.clear()
    }
}

impl<T, OS, AS> BatchUpdate<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: BatchUpdate<usize, usize>,
    OS: FiniteSpace + Clone,
    AS: FiniteSpace + Clone,
    // Compiler bug https://github.com/rust-lang/rust/issues/85451
    T::HistoryBuffer: 'static,
    OS: 'static,
    AS: 'static,
{
    type HistoryBuffer = FiniteSpaceBuffer<T::HistoryBuffer, OS, AS>;

    fn new_buffer(&self) -> Self::HistoryBuffer {
        FiniteSpaceBuffer {
            buffer: self.agent.new_buffer(),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn TimeSeriesLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        self.agent
            .batch_update(buffers.into_iter().map(|b| &mut b.buffer), logger)
    }
}

impl<'a, T, OS, AS> MakeActor<'a, OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: MakeActor<'a, usize, usize>,
    OS: FiniteSpace + Sync + 'a,
    AS: FiniteSpace + Sync + 'a,
{
    type Actor = FiniteSpaceAgent<T::Actor, &'a OS, &'a AS>;

    fn make_actor(&'a self, seed: u64) -> Self::Actor {
        FiniteSpaceAgent {
            agent: self.agent.make_actor(seed),
            observation_space: &self.observation_space,
            action_space: &self.action_space,
        }
    }
}

/// Convert a finite-space step into an index step.
fn indexed_step<OS, AS>(
    step: &TransientStep<OS::Element, AS::Element>,
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
        next: step.next.as_ref().map(|o| observation_space.to_index(o)),
    }
}

/// Convert a finite-space `PartialStep` into an index step.
fn indexed_partial_step<OS, AS>(
    step: &PartialStep<OS::Element, AS::Element>,
    observation_space: &OS,
    action_space: &AS,
) -> PartialStep<usize, usize>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    Step {
        observation: observation_space.to_index(&step.observation),
        action: action_space.to_index(&step.action),
        reward: step.reward,
        next: match &step.next {
            Successor::Continue(()) => Successor::Continue(()),
            Successor::Terminate => Successor::Terminate,
            Successor::Interrupt(s) => Successor::Interrupt(observation_space.to_index(s)),
        },
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
