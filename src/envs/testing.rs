//! Environment testing utilities
use super::as_stateful::AsStateful;
use super::{EnvStructure, Environment, StatefulEnvironment};
use crate::agents::RandomAgent;
use crate::simulator;
use crate::spaces::Space;

/// Run a stateless environment and check that invariants are satisfied.
pub fn run_stateless<E: Environment>(env: E, num_steps: u64, seed: u64) {
    run_stateful(&mut env.as_stateful(seed), num_steps, seed + 1)
}

/// Run a stateful environment and check that invariants are satisfied.
pub fn run_stateful<E: StatefulEnvironment>(env: &mut E, mut num_steps: u64, seed: u64) {
    let EnvStructure {
        observation_space,
        action_space,
        reward_range: (min_reward, max_reward),
        discount_factor,
    } = env.structure();
    assert!(discount_factor >= 0.0);
    assert!(discount_factor <= 1.0);

    let mut agent = RandomAgent::new(action_space, seed);
    simulator::run(env, &mut agent, &mut |step| {
        assert!(step.reward >= min_reward);
        assert!(step.reward <= max_reward);
        if let Some(obs) = step.next_observation {
            assert!(observation_space.contains(obs))
        } else {
            assert!(step.episode_done);
        }

        num_steps -= 1;
        num_steps > 0
    });
}
