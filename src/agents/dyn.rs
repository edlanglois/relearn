//! Dynamic agent
use super::{
    Actor, ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError, HistoryDataBound,
    WriteExperience,
};
use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::spaces::Space;
use crate::Prng;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// Trait alias for a dynamic agent
pub trait DynAgent<O, A>:
    Agent<
    O,
    A,
    Actor = Box<dyn Actor<O, A, EpisodeState = Box<dyn Any>>>,
    HistoryBuffer = Box<dyn AnyWriteExperience<O, A>>,
>
{
}
impl<T, O, A> DynAgent<O, A> for T where
    T: Agent<
        O,
        A,
        Actor = Box<dyn Actor<O, A, EpisodeState = Box<dyn Any>>>,
        HistoryBuffer = Box<dyn AnyWriteExperience<O, A>>,
    >
{
}

/// Agent that boxes all associated types.
///
/// The `HistoryBuffer`, `Actor`, and actor's `EpisodeState` are all boxed.
/// Implements [`DynAgent`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoxAgent<T>(pub T);

impl<T> BoxAgent<T> {
    pub const fn new(agent: T) -> Self {
        Self(agent)
    }
}

impl<T, OS, AS> BuildAgent<OS, AS> for BoxAgent<T>
where
    T: BuildAgent<OS, AS>,
    T::Agent: 'static,
    OS: Space,
    OS::Element: 'static,
    AS: Space,
    AS::Element: 'static,
{
    type Agent = BoxAgent<T::Agent>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        rng: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(BoxAgent(self.0.build_agent(env, rng)?))
    }
}

impl<T, O, A> Agent<O, A> for BoxAgent<T>
where
    T: Agent<O, A>,
    T::HistoryBuffer: 'static,
    T::Actor: 'static,
    <T::Actor as Actor<O, A>>::EpisodeState: 'static,
    O: 'static,
    A: 'static,
{
    type Actor = Box<dyn Actor<O, A, EpisodeState = Box<dyn Any>>>;

    fn actor(&self, mode: ActorMode) -> Self::Actor {
        Box::new(BoxActor(self.0.actor(mode)))
    }
}

/// Subtrait of `WriteExperience` and `Any`.
///
/// See <https://users.rust-lang.org/t/why-does-downcasting-not-work-for-subtraits/33286>
/// for a discussion of why the casting is so complicated in this case.
pub trait AnyWriteExperience<O, A>: WriteExperience<O, A> + Any {
    /// Upcast to a mutable `dyn Any` reference
    fn upcast_any_mut(&mut self) -> &mut dyn Any;
}
impl<T, O, A> AnyWriteExperience<O, A> for T
where
    T: WriteExperience<O, A> + Any,
{
    #[inline]
    fn upcast_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl<O, A> dyn AnyWriteExperience<O, A> {
    /// Downcast to a mutable reference
    fn downcast_mut<U: 'static>(&mut self) -> Option<&mut U> {
        self.upcast_any_mut().downcast_mut()
    }
}

impl<T, O, A> BatchUpdate<O, A> for BoxAgent<T>
where
    T: BatchUpdate<O, A>,
    T::HistoryBuffer: Any + 'static,
    O: 'static,
    A: 'static,
{
    type HistoryBuffer = Box<dyn AnyWriteExperience<O, A>>;

    fn buffer(&self) -> Self::HistoryBuffer {
        Box::new(self.0.buffer())
    }

    fn min_update_size(&self) -> HistoryDataBound {
        self.0.min_update_size()
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        self.0.batch_update(
            buffers.into_iter().map(|b| {
                b.as_mut()
                    .downcast_mut()
                    .expect("buffer has incorrect dynamic type")
            }),
            logger,
        )
    }

    fn batch_update_single(
        &mut self,
        buffer: &mut Self::HistoryBuffer,
        logger: &mut dyn StatsLogger,
    ) {
        self.0.batch_update_single(
            buffer
                .as_mut()
                .downcast_mut()
                .expect("buffer has incorrect dynamic type"),
            logger,
        )
    }

    fn batch_update_slice(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn StatsLogger,
    ) {
        self.batch_update(buffers, logger)
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoxActor<T>(pub T);
impl<T, O, A> Actor<O, A> for BoxActor<T>
where
    T: Actor<O, A>,
    T::EpisodeState: 'static,
{
    type EpisodeState = Box<dyn Any>;

    fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        Box::new(self.0.new_episode_state(rng))
    }

    fn act(&self, episode_state: &mut Self::EpisodeState, observation: &O, rng: &mut Prng) -> A {
        self.0.act(
            episode_state
                .downcast_mut()
                .expect("episode_state has incorrect dynamic type"),
            observation,
            rng,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::{testing, TabularQLearningAgentConfig};
    use super::*;

    #[test]
    fn box_agent_learns_determinstic_bandit() {
        testing::train_deterministic_bandit(
            &BoxAgent::new(TabularQLearningAgentConfig::default()),
            1000,
            0.9,
        );
    }

    /// Wraps an agent configuration to build a boxed `DynAgent`.
    pub struct BuildDynAgent<T>(T);
    impl<T, OS, AS> BuildAgent<OS, AS> for BuildDynAgent<T>
    where
        T: BuildAgent<OS, AS>,
        T::Agent: 'static,
        OS: Space + 'static,
        AS: Space + 'static,
    {
        type Agent = Box<dyn DynAgent<OS::Element, AS::Element>>;

        fn build_agent(
            &self,
            env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
            rng: &mut Prng,
        ) -> Result<Self::Agent, BuildAgentError> {
            Ok(Box::new(BoxAgent(self.0.build_agent(env, rng)?)))
        }
    }

    #[test]
    /// Check that `BoxAgent` implements `DynAgent` and that training a `DynAgent` succeeds.
    fn box_agent_as_dyn_agent_learns_determinstic_bandit() {
        testing::train_deterministic_bandit(
            &BuildDynAgent(TabularQLearningAgentConfig::default()),
            1000,
            0.9,
        );
    }
}
