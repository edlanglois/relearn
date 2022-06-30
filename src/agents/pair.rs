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

/// A pair of agents / actors / configs for a two-agent environment.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentPair<T0, T1>(pub T0, pub T1);

impl<T0, T1, O0, O1, A0, A1> Agent<(O0, O1), (A0, A1)> for AgentPair<T0, T1>
where
    T0: Agent<O0, A0>,
    T1: Agent<O1, A1>,
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

    fn initial_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        (self.0.initial_state(rng), self.1.initial_state(rng))
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
    type Feedback = (T0::Feedback, T1::Feedback);
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
        let (buffers1, buffers2): (Vec<_>, Vec<_>) =
            buffers.into_iter().map(|b| (&mut b.0, &mut b.1)).unzip();
        self.0.batch_update(buffers1.into_iter(), logger);
        self.1.batch_update(buffers2.into_iter(), logger);
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HistoryBufferPair<B0, B1>(pub B0, pub B1);

impl<B0, B1, O0, O1, A0, A1, F0, F1> WriteExperience<(O0, O1), (A0, A1), (F0, F1)>
    for HistoryBufferPair<B0, B1>
where
    B0: WriteExperience<O0, A0, F0>,
    B1: WriteExperience<O1, A1, F1>,
{
}

impl<B0, B1, O0, O1, A0, A1, F0, F1> WriteExperienceIncremental<(O0, O1), (A0, A1), (F0, F1)>
    for HistoryBufferPair<B0, B1>
where
    B0: WriteExperienceIncremental<O0, A0, F0>,
    B1: WriteExperienceIncremental<O1, A1, F1>,
{
    fn write_step(
        &mut self,
        step: PartialStep<(O0, O1), (A0, A1), (F0, F1)>,
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
fn split_partial_step<O0, O1, A0, A1, F0, F1>(
    step: PartialStep<(O0, O1), (A0, A1), (F0, F1)>,
) -> (PartialStep<O0, A0, F0>, PartialStep<O1, A1, F1>) {
    let (o0, o1) = step.observation;
    let (a0, a1) = step.action;
    let (f0, f1) = step.feedback;
    let (n0, n1) = match step.next {
        Successor::Continue(()) => (Successor::Continue(()), Successor::Continue(())),
        Successor::Terminate => (Successor::Terminate, Successor::Terminate),
        Successor::Interrupt((no0, no1)) => (Successor::Interrupt(no0), Successor::Interrupt(no1)),
    };
    let step0 = PartialStep {
        observation: o0,
        action: a0,
        feedback: f0,
        next: n0,
    };
    let step1 = PartialStep {
        observation: o1,
        action: a1,
        feedback: f1,
        next: n1,
    };
    (step0, step1)
}

impl<T0, T1, OS0, OS1, AS0, AS1, FS0, FS1>
    BuildAgent<TupleSpace2<OS0, OS1>, TupleSpace2<AS0, AS1>, TupleSpace2<FS0, FS1>>
    for AgentPair<T0, T1>
where
    T0: BuildAgent<OS0, AS0, FS0> + 'static,
    T1: BuildAgent<OS1, AS1, FS1> + 'static,
    OS0: Space + Clone + 'static,
    OS1: Space + Clone + 'static,
    AS0: Space + Clone + 'static,
    AS1: Space + Clone + 'static,
    FS0: Space + Clone + 'static,
    FS1: Space + Clone + 'static,
{
    type Agent = AgentPair<T0::Agent, T1::Agent>;
    fn build_agent(
        &self,
        env: &dyn EnvStructure<
            ObservationSpace = TupleSpace2<OS0, OS1>,
            ActionSpace = TupleSpace2<AS0, AS1>,
            FeedbackSpace = TupleSpace2<FS0, FS1>,
        >,
        rng: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        let (env1, env2) = split_env_structure(env);
        let agent1 = self.0.build_agent(&env1, rng)?;
        let agent2 = self.1.build_agent(&env2, rng)?;
        Ok(AgentPair(agent1, agent2))
    }
}

fn split_env_structure<OS0, OS1, AS0, AS1, FS0, FS1>(
    env: &dyn EnvStructure<
        ObservationSpace = TupleSpace2<OS0, OS1>,
        ActionSpace = TupleSpace2<AS0, AS1>,
        FeedbackSpace = TupleSpace2<FS0, FS1>,
    >,
) -> (
    StoredEnvStructure<OS0, AS0, FS0>,
    StoredEnvStructure<OS1, AS1, FS1>,
)
where
    OS0: Space,
    OS1: Space,
    AS0: Space,
    AS1: Space,
    FS0: Space,
    FS1: Space,
{
    let TupleSpace2(os0, os1) = env.observation_space();
    let TupleSpace2(as0, as1) = env.action_space();
    let TupleSpace2(fs0, fs1) = env.feedback_space();
    let discount_factor = env.discount_factor();
    (
        StoredEnvStructure::new(os0, as0, fs0, discount_factor),
        StoredEnvStructure::new(os1, as1, fs1, discount_factor),
    )
}

// TODO tests
