//! Environment testing utilities
use super::as_stateful::AsStateful;
use super::{EnvStructure, Environment, StatefulEnvironment};
use crate::agents::RandomAgent;
use crate::simulation;
use crate::spaces::Space;
use std::fmt::Debug;

/// Run a stateless environment and check that invariants are satisfied.
pub fn run_stateless<E>(env: E, num_steps: u64, seed: u64)
where
    E: Environment,
    <E as Environment>::ObservationSpace: Debug,
    <<E as Environment>::ObservationSpace as Space>::Element: Debug,
    <E as Environment>::ActionSpace: Debug,
    <<E as Environment>::ActionSpace as Space>::Element: Debug,
{
    run_stateful(&mut env.as_stateful(seed), num_steps, seed + 1)
}

/// Run a stateful environment and check that invariants are satisfied.
pub fn run_stateful<E>(env: &mut E, mut num_steps: u64, seed: u64)
where
    E: StatefulEnvironment,
    <E as StatefulEnvironment>::ObservationSpace: Debug,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Debug,
    <E as StatefulEnvironment>::ActionSpace: Debug,
    <<E as StatefulEnvironment>::ActionSpace as Space>::Element: Debug,
{
    let EnvStructure {
        observation_space,
        action_space,
        reward_range: (min_reward, max_reward),
        discount_factor,
    } = env.structure();
    assert!(discount_factor >= 0.0);
    assert!(discount_factor <= 1.0);

    let mut agent = RandomAgent::new(action_space, seed);
    simulation::run(env, &mut agent, |step| {
        assert!(step.reward >= min_reward);
        assert!(step.reward <= max_reward);
        if let Some(obs) = step.next_observation {
            assert!(observation_space.contains(obs))
        } else {
            assert!(step.episode_done);
        }

        num_steps -= 1;
        num_steps == 0
    });
}
