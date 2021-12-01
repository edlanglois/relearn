use super::{Actor, ActorMode, BuildAgent, BuildAgentError, SetActorMode, SynchronousAgent};
use crate::envs::{EnvStructure, Successor};
use crate::logging::TimeSeriesLogger;
use crate::simulation::{Step, TransientStep};
use crate::spaces::{FiniteSpace, IndexSpace, Space};

/// Build an agent for finite, indexed action and observation spaces.
///
/// This is a helper trait that automatically implements [`BuildAgent`]
/// with all finite-space environments.
pub trait BuildIndexAgent {
    type Agent: SynchronousAgent<<IndexSpace as Space>::Element, <IndexSpace as Space>::Element>
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
            .expect("Invalid action index")
    }

    fn reset(&mut self) {
        self.agent.reset()
    }
}

impl<T, OS, AS> SynchronousAgent<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: SynchronousAgent<usize, usize>, // Index-space agent
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

/*
impl<T, OS, AS> BatchUpdate<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: OffPolicyAgent<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn batch_update(
        &mut self,
        history: &mut dyn HistoryBuffer<OS::Element, AS::Element>,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        logger.start_event(Event::AgentOptPeriod).unwrap();

        let mut steps = history.steps().peekable();
        while let Some(step) = steps.next() {
            let next_observation_index = if matches!(step.next, Successor::Continue(())) {
                match steps.peek() {
                    Some(next_step) => {
                        Some(self.observation_space.to_index(&next_step.observation))
                    }
                    None => break, // incomplete step
                }
            } else {
                None
            };
            let ref_index_next = match &step.next {
                Successor::Continue(()) => {
                    Successor::Continue(next_observation_index.as_ref().unwrap())
                }
                Successor::Terminate => Successor::Terminate,
                Successor::Interrupt(obs) => {
                    Successor::Interrupt(self.observation_space.to_index(obs))
                }
            };
            let transient_step = TransientStep {
                observation: self.observation_space.to_index(&step.observation),
                action: self.action_space.to_index(&step.action),
                reward: step.reward,
                next: ref_index_next,
            };
            self.agent.update(transient_step, logger)
        }

        logger.end_event(Event::AgentOptPeriod).unwrap();
    }
}
*/

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
