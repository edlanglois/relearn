use super::{
    Actor, ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError, HistoryDataBound,
    WriteExperience, WriteExperienceError, WriteExperienceIncremental,
};
use crate::envs::{EnvStructure, StoredEnvStructure, Successor};
use crate::logging::StatsLogger;
use crate::simulation::PartialStep;
use crate::spaces::{Space, TupleSpace2};
use crate::Prng;
use serde::{Deserialize, Serialize};
use std::iter;

/// A pair of agents / actors / configs for a two-agent environment.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentPair<T0, T1>(pub T0, pub T1);

impl<T0, T1, O0, O1, A0, A1> Agent<(O0, O1), (A0, A1)> for AgentPair<T0, T1>
where
    T0: Agent<O0, A0>,
    T1: Agent<O1, A1>,
    // Compiler bug <https://github.com/rust-lang/rust/issues/85451>
    T0::HistoryBuffer: 'static,
    T1::HistoryBuffer: 'static,
{
    type Actor = AgentPair<T0::Actor, T1::Actor>;

    fn actor(&self, mode: ActorMode) -> Self::Actor {
        AgentPair(self.0.actor(mode), self.1.actor(mode))
    }
}

impl<T0, T1, O0, O1, A0, A1> Actor<(O0, O1), (A0, A1)> for AgentPair<T0, T1>
where
    T0: Actor<O0, A0>,
    T1: Actor<O1, A1>,
{
    type EpisodeState = (T0::EpisodeState, T1::EpisodeState);

    fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        (self.0.new_episode_state(rng), self.1.new_episode_state(rng))
    }

    fn act(
        &self,
        episode_state: &mut Self::EpisodeState,
        observation: &(O0, O1),
        rng: &mut Prng,
    ) -> (A0, A1) {
        (
            self.0.act(&mut episode_state.0, &observation.0, rng),
            self.1.act(&mut episode_state.1, &observation.1, rng),
        )
    }
}

impl<T0, T1, O0, O1, A0, A1> BatchUpdate<(O0, O1), (A0, A1)> for AgentPair<T0, T1>
where
    T0: BatchUpdate<O0, A0>,
    T1: BatchUpdate<O1, A1>,
    // Compiler bug <https://github.com/rust-lang/rust/issues/85451>
    T0::HistoryBuffer: 'static,
    T1::HistoryBuffer: 'static,
{
    type HistoryBuffer = HistoryBufferPair<T0::HistoryBuffer, T1::HistoryBuffer>;

    fn buffer(&self) -> Self::HistoryBuffer {
        HistoryBufferPair(self.0.buffer(), self.1.buffer())
    }

    fn min_update_size(&self) -> HistoryDataBound {
        self.0.min_update_size().max(self.1.min_update_size())
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        let mut buffers1 = Vec::new();
        let mut buffers2 = Vec::new();
        for buffer in buffers {
            buffers1.push(&mut buffer.0);
            buffers2.push(&mut buffer.1);
        }
        self.0.batch_update(buffers1.into_iter(), logger);
        self.1.batch_update(buffers2.into_iter(), logger);
    }

    fn batch_update_single(
        &mut self,
        buffer: &mut Self::HistoryBuffer,
        logger: &mut dyn StatsLogger,
    ) {
        self.batch_update(iter::once(buffer), logger)
    }

    fn batch_update_slice(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn StatsLogger,
    ) {
        self.batch_update(buffers, logger)
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HistoryBufferPair<B0, B1>(pub B0, pub B1);

impl<B0, B1, O0, O1, A0, A1> WriteExperience<(O0, O1), (A0, A1)> for HistoryBufferPair<B0, B1>
where
    B0: WriteExperience<O0, A0>,
    B1: WriteExperience<O1, A1>,
{
}

impl<B0, B1, O0, O1, A0, A1> WriteExperienceIncremental<(O0, O1), (A0, A1)>
    for HistoryBufferPair<B0, B1>
where
    B0: WriteExperienceIncremental<O0, A0>,
    B1: WriteExperienceIncremental<O1, A1>,
{
    fn write_step(
        &mut self,
        step: PartialStep<(O0, O1), (A0, A1)>,
    ) -> Result<(), WriteExperienceError> {
        let (step1, step2) = split_partial_step(step);
        self.0.write_step(step1)?;
        self.1.write_step(step2)
    }
    fn end_experience(&mut self) {
        self.0.end_experience();
        self.1.end_experience();
    }
}

#[allow(clippy::missing_const_for_fn)]
fn split_partial_step<O1, O2, A1, A2>(
    step: PartialStep<(O1, O2), (A1, A2)>,
) -> (PartialStep<O1, A1>, PartialStep<O2, A2>) {
    let (o1, o2) = step.observation;
    let (a1, a2) = step.action;
    let (n1, n2) = match step.next {
        Successor::Continue(()) => (Successor::Continue(()), Successor::Continue(())),
        Successor::Terminate => (Successor::Terminate, Successor::Terminate),
        Successor::Interrupt((no1, no2)) => (Successor::Interrupt(no1), Successor::Interrupt(no2)),
    };
    let step1 = PartialStep {
        observation: o1,
        action: a1,
        reward: step.reward,
        next: n1,
    };
    let step2 = PartialStep {
        observation: o2,
        action: a2,
        reward: step.reward,
        next: n2,
    };
    (step1, step2)
}

impl<T0, T1, OS0, OS1, AS0, AS1> BuildAgent<TupleSpace2<OS0, OS1>, TupleSpace2<AS0, AS1>>
    for AgentPair<T0, T1>
where
    T0: BuildAgent<OS0, AS0> + 'static,
    T1: BuildAgent<OS1, AS1> + 'static,
    OS0: Space + Clone + 'static,
    OS1: Space + Clone + 'static,
    AS0: Space + Clone + 'static,
    AS1: Space + Clone + 'static,
{
    type Agent = AgentPair<T0::Agent, T1::Agent>;
    fn build_agent(
        &self,
        env: &dyn EnvStructure<
            ObservationSpace = TupleSpace2<OS0, OS1>,
            ActionSpace = TupleSpace2<AS0, AS1>,
        >,
        rng: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        let (env1, env2) = split_env_structure(env);
        let agent1 = self.0.build_agent(&env1, rng)?;
        let agent2 = self.1.build_agent(&env2, rng)?;
        Ok(AgentPair(agent1, agent2))
    }
}

fn split_env_structure<OS1, OS2, AS1, AS2>(
    env: &dyn EnvStructure<
        ObservationSpace = TupleSpace2<OS1, OS2>,
        ActionSpace = TupleSpace2<AS1, AS2>,
    >,
) -> (StoredEnvStructure<OS1, AS1>, StoredEnvStructure<OS2, AS2>)
where
    OS1: Space,
    OS2: Space,
    AS1: Space,
    AS2: Space,
{
    let TupleSpace2(os1, os2) = env.observation_space();
    let TupleSpace2(as1, as2) = env.action_space();
    let reward_range = env.reward_range();
    let discount_factor = env.discount_factor();
    (
        StoredEnvStructure::new(os1, as1, reward_range, discount_factor),
        StoredEnvStructure::new(os2, as2, reward_range, discount_factor),
    )
}

// TODO tests
