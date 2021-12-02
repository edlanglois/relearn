use super::{
    Actor, ActorMode, BatchUpdate, BuildAgent, BuildAgentError, BuildBatchAgent, MakeActor,
    PureActor, SetActorMode, SynchronousUpdate,
};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::simulation::TransientStep;
use crate::spaces::Space;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Configuration for [`PureAsActor`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PureAsActorConfig<TC> {
    pub actor_config: TC,
}

impl<TC> PureAsActorConfig<TC> {
    pub const fn new(actor_config: TC) -> Self {
        Self { actor_config }
    }
}

impl<TC, OS, AS> BuildAgent<OS, AS> for PureAsActorConfig<TC>
where
    TC: BuildAgent<OS, AS>,
    TC::Agent: PureActor<OS::Element, AS::Element>,
    OS: Space,
    AS: Space,
{
    type Agent = PureAsActor<TC::Agent, OS::Element, AS::Element>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(PureAsActor::new(
            self.actor_config.build_agent(env, rng.gen())?,
            rng.gen(),
        ))
    }
}

impl<TC, OS, AS> BuildBatchAgent<OS, AS> for PureAsActorConfig<TC>
where
    TC: BuildBatchAgent<OS, AS>,
    TC::BatchAgent: PureActor<OS::Element, AS::Element>,
    OS: Space,
    AS: Space,
{
    type BatchAgent = PureAsActor<TC::BatchAgent, OS::Element, AS::Element>;

    fn build_batch_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchAgent, BuildAgentError> {
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(PureAsActor::new(
            self.actor_config.build_batch_agent(env, rng.gen())?,
            rng.gen(),
        ))
    }
}

/// Wrapper that implements [`Actor`] for [`PureActor`] by storing internal state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureAsActor<T, O, A>
where
    T: PureActor<O, A>,
{
    pub actor: T,
    pub state: T::State,
    rng: StdRng,
}

impl<T, O, A> PureAsActor<T, O, A>
where
    T: PureActor<O, A>,
{
    pub fn new(actor: T, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            state: actor.initial_state(rng.gen()),
            actor,
            rng,
        }
    }
}

impl<T, O, A> Actor<O, A> for PureAsActor<T, O, A>
where
    T: PureActor<O, A>,
{
    fn act(&mut self, observation: &O) -> A {
        self.actor.act(&mut self.state, observation)
    }

    fn reset(&mut self) {
        self.actor.reset_state(&mut self.state);
    }
}

impl<T, O, A> SynchronousUpdate<O, A> for PureAsActor<T, O, A>
where
    T: PureActor<O, A> + SynchronousUpdate<O, A>,
{
    fn update(&mut self, step: TransientStep<O, A>, logger: &mut dyn TimeSeriesLogger) {
        self.actor.update(step, logger)
    }
}

impl<T, O, A> BatchUpdate<O, A> for PureAsActor<T, O, A>
where
    T: PureActor<O, A> + BatchUpdate<O, A>,
{
    type HistoryBuffer = T::HistoryBuffer;

    fn new_buffer(&self) -> Self::HistoryBuffer {
        self.actor.new_buffer()
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn TimeSeriesLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        self.actor.batch_update(buffers, logger);
    }
}

impl<'a, T, O, A> MakeActor<'a, O, A> for PureAsActor<T, O, A>
where
    T: PureActor<O, A> + Sync + 'a,
    T::State: Send,
{
    type Actor = PureAsActor<&'a T, O, A>;

    fn make_actor(&'a self, seed: u64) -> Self::Actor {
        PureAsActor::new(&self.actor, seed)
    }
}

impl<T, O, A> SetActorMode for PureAsActor<T, O, A>
where
    T: PureActor<O, A> + SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.actor.set_actor_mode(mode)
    }
}
