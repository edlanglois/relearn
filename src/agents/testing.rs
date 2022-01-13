//! Agent testing utilities
use crate::agents::{Actor, ActorMode, Agent, BuildAgent};
use crate::envs::{DeterministicBandit, Environment};
use crate::simulation;
use crate::spaces::{IndexSpace, SingletonSpace};
use crate::Prng;
use rand::SeedableRng;

/// Check that the agent can be trained to perform well on a trivial bandit environment.
///
/// The environment is a deterministic multi-armed bandit with two arms:
/// the first arm always gives 0 reward and the second 1.
pub fn train_deterministic_bandit<TC>(agent_config: &TC, num_periods: usize, threshold: f64)
where
    TC: BuildAgent<SingletonSpace, IndexSpace>,
{
    let mut env_rng = Prng::seed_from_u64(18);
    let mut agent_rng = Prng::seed_from_u64(19);

    let env = DeterministicBandit::from_values(vec![0.0, 1.0]);
    let mut agent = agent_config
        .build_agent(&env, &mut agent_rng)
        .expect("failed to build agent");

    // Training
    simulation::train_serial(
        &mut agent,
        &env,
        num_periods,
        &mut env_rng,
        &mut agent_rng,
        &mut (),
    );

    eval_deterministic_bandit(agent.actor(ActorMode::Evaluation), &env, threshold);
}

/// Evaluate a trained agent on the 0-1 deterministic bandit environment.
#[allow(clippy::cast_possible_truncation)]
pub fn eval_deterministic_bandit<T>(actor: T, env: &DeterministicBandit, threshold: f64)
where
    T: Actor<(), usize>,
{
    let num_eval_steps = 1000;
    let mut action_1_count = 0;
    for step in env.run(actor, 44, ()).take(num_eval_steps) {
        if step.action == 1 {
            action_1_count += 1;
        }
    }

    assert!(action_1_count >= ((num_eval_steps as f64) * threshold) as u64);
}
