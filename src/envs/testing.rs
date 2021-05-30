//! Environment testing utilities
use super::{EnvStructure, Environment, StatefulEnvironment, WithState};
use crate::agents::{RandomAgent, Step};
use crate::simulation;
use crate::simulation::hooks::{ClosureHook, StepLimit};
use crate::spaces::{SampleSpace, Space};
use std::fmt::Debug;

/// Run a stateless environment and check that invariants are satisfied.
pub fn run_stateless<E>(env: E, num_steps: u64, seed: u64)
where
    E: Environment,
    <E as EnvStructure>::ObservationSpace: Debug,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Debug + Clone,
    <E as EnvStructure>::ActionSpace: Debug + SampleSpace,
    <<E as EnvStructure>::ActionSpace as Space>::Element: Debug,
{
    run_stateful(&mut env.with_state(seed), num_steps, seed + 1)
}

/// Run a stateful environment and check that invariants are satisfied.
pub fn run_stateful<E>(env: &mut E, num_steps: u64, seed: u64)
where
    E: StatefulEnvironment,
    <E as EnvStructure>::ObservationSpace: Debug,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Debug + Clone,
    <E as EnvStructure>::ActionSpace: Debug + SampleSpace,
    <<E as EnvStructure>::ActionSpace as Space>::Element: Debug,
{
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let (min_reward, max_reward) = env.reward_range();
    assert!(min_reward <= max_reward);

    let discount_factor = env.discount_factor();
    assert!(discount_factor >= 0.0);
    assert!(discount_factor <= 1.0);

    if num_steps == 0 {
        return;
    }

    let mut agent = RandomAgent::new(action_space, seed);
    simulation::run_agent(
        env,
        &mut agent,
        &mut (),
        &mut (
            ClosureHook::from(|step: &Step<_, _>| -> bool {
                assert!(step.reward >= min_reward);
                assert!(step.reward <= max_reward);
                if let Some(obs) = &step.next_observation {
                    assert!(observation_space.contains(obs))
                } else {
                    assert!(step.episode_done);
                }
                true
            }),
            StepLimit::new(num_steps),
        ),
    );
}
