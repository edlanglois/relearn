//! Agent testing utilities
use crate::agents::{Actor, ActorMode, PureActor, PureAsActor, SetActorMode, SynchronousUpdate};
use crate::envs::{DeterministicBandit, EnvStructure, Environment, IntoEnv, PomdpEnv};
use crate::simulation;
use crate::simulation::hooks::{IndexedActionCounter, StepLimit};

/// Check that the agent can be trained to perform well on a trivial bandit environment.
///
/// The environment is a deterministic multi-armed bandit with two arms:
/// the first arm always gives 0 reward and the second 1.
pub fn train_deterministic_bandit<A, F>(make_agent: F, num_train_steps: u64, threshold: f64)
where
    A: Actor<(), usize> + SynchronousUpdate<(), usize> + SetActorMode,
    F: FnOnce(&PomdpEnv<DeterministicBandit>) -> A,
{
    let mut env = DeterministicBandit::from_values(vec![0.0, 1.0]).into_env(0);
    let mut agent = make_agent(&env);

    // Training
    if num_train_steps > 0 {
        simulation::run_agent(&mut env, &mut agent, StepLimit::new(num_train_steps), ());
    }

    eval_deterministic_bandit(agent, &mut env, threshold);
}

/// Check that the pure agent can be trained to perform well on a trivial bandit environment.
///
/// The environment is a deterministic multi-armed bandit with two arms:
/// the first arm always gives 0 reward and the second 1.
pub fn pure_train_deterministic_bandit<A, F>(make_agent: F, num_train_steps: u64, threshold: f64)
where
    A: PureActor<(), usize> + SynchronousUpdate<(), usize> + SetActorMode,
    F: FnOnce(&PomdpEnv<DeterministicBandit>) -> A,
{
    train_deterministic_bandit(
        |env| PureAsActor::new(make_agent(env), 1),
        num_train_steps,
        threshold,
    )
}

/// Evaluate a trained agent on the 0-1 deterministic bandit environment.
#[allow(clippy::cast_possible_truncation)]
pub fn eval_deterministic_bandit<T>(
    mut agent: T,
    env: &mut PomdpEnv<DeterministicBandit>,
    threshold: f64,
) where
    T: Actor<(), usize> + SynchronousUpdate<(), usize> + SetActorMode,
{
    // Evaluation
    agent.set_actor_mode(ActorMode::Release);

    let num_eval_steps = 1000;
    let mut action_counter = IndexedActionCounter::new(env.action_space());
    env.run(agent, ())
        .take(num_eval_steps)
        .for_each(|s| action_counter.call_(&s));
    let action_1_count = action_counter.counts[1];
    assert!(action_1_count >= ((num_eval_steps as f64) * threshold) as u64);
}
