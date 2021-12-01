use super::{Actor, ActorMode, BuildAgent, BuildAgentError, SetActorMode, SynchronousUpdate};
use crate::envs::{EnvStructure, StoredEnvStructure, Successor};
use crate::logging::TimeSeriesLogger;
use crate::simulation::TransientStep;
use crate::spaces::{Space, TupleSpace2};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A pair of agents for a two-agent environment.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AgentPair<T, U>(pub T, pub U);

impl<T, U, O1, O2, A1, A2> Actor<(O1, O2), (A1, A2)> for AgentPair<T, U>
where
    T: Actor<O1, A1>,
    U: Actor<O2, A2>,
{
    fn act(&mut self, observation: &(O1, O2)) -> (A1, A2) {
        (self.0.act(&observation.0), self.1.act(&observation.1))
    }

    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
}

impl<T, U, O1, O2, A1, A2> SynchronousUpdate<(O1, O2), (A1, A2)> for AgentPair<T, U>
where
    T: SynchronousUpdate<O1, A1>,
    U: SynchronousUpdate<O2, A2>,
{
    fn update(
        &mut self,
        step: TransientStep<(O1, O2), (A1, A2)>,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        let (o1, o2) = step.observation;
        let (a1, a2) = step.action;
        let (n1, n2) = match step.next {
            Successor::Continue((no1, no2)) => (Successor::Continue(no1), Successor::Continue(no2)),
            Successor::Terminate => (Successor::Terminate, Successor::Terminate),
            Successor::Interrupt((no1, no2)) => {
                (Successor::Interrupt(no1), Successor::Interrupt(no2))
            }
        };
        self.0.update(
            TransientStep {
                observation: o1,
                action: a1,
                reward: step.reward,
                next: n1,
            },
            logger,
        );
        self.1.update(
            TransientStep {
                observation: o2,
                action: a2,
                reward: step.reward,
                next: n2,
            },
            logger,
        );
    }
}

impl<T, U> SetActorMode for AgentPair<T, U>
where
    T: SetActorMode,
    U: SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.0.set_actor_mode(mode);
        self.1.set_actor_mode(mode);
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

impl<TB, UB, OS1, OS2, AS1, AS2> BuildAgent<TupleSpace2<OS1, OS2>, TupleSpace2<AS1, AS2>>
    for AgentPair<TB, UB>
where
    TB: BuildAgent<OS1, AS1>,
    UB: BuildAgent<OS2, AS2>,
    OS1: Space + Clone,
    OS2: Space + Clone,
    AS1: Space + Clone,
    AS2: Space + Clone,
{
    type Agent = AgentPair<TB::Agent, UB::Agent>;
    fn build_agent(
        &self,
        env: &dyn EnvStructure<
            ObservationSpace = TupleSpace2<OS1, OS2>,
            ActionSpace = TupleSpace2<AS1, AS2>,
        >,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        let mut seed_rng = StdRng::seed_from_u64(seed);
        let (env1, env2) = split_env_structure(env);
        let agent1 = self.0.build_agent(&env1, seed_rng.gen())?;
        let agent2 = self.1.build_agent(&env2, seed_rng.gen())?;
        Ok(AgentPair(agent1, agent2))
    }
}

// Need to implement BatchUpdate for AgentPair first
/*
impl<TB, UB, OS1, OS2, AS1, AS2> BuildBatchUpdateActor<TupleSpace2<OS1, OS2>, TupleSpace2<AS1, AS2>>
    for AgentPair<TB, UB>
where
    TB: BuildBatchUpdateActor<OS1, AS1>,
    UB: BuildBatchUpdateActor<OS2, AS2>,
    OS1: Space + Clone,
    OS2: Space + Clone,
    AS1: Space + Clone,
    AS2: Space + Clone,
{
    type BatchUpdateActor = AgentPair<TB::BatchUpdateActor, UB::BatchUpdateActor>;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<
            ObservationSpace = TupleSpace2<OS1, OS2>,
            ActionSpace = TupleSpace2<AS1, AS2>,
        >,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        let mut seed_rng = StdRng::seed_from_u64(seed);
        let (env1, env2) = split_env_structure(env);
        let actor1 = self.0.build_batch_update_actor(&env1, seed_rng.gen())?;
        let actor1 = self.1.build_batch_update_actor(&env2, seed_rng.gen())?;
        Ok(AgentPair(actor1, actor2))
    }
}
*/
