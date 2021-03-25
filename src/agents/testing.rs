//! Agent testing utilities
use crate::envs::{AsStateful, DeterministicBandit, StatefulEnvironment};
use crate::simulation;
use crate::spaces::{IndexSpace, SingletonSpace};
use crate::{Agent, EnvStructure};

/// Check that the agent can be trained to perform well on a trivial bandit environment.
///
/// The environment is a deterministic multi-armed bandit with two arms:
/// the first arm always gives 0 reward and the second 1.
pub fn train_deterministic_bandit<A, F>(make_agent: F, num_train_steps: u64, threshold: f64)
where
    A: Agent<(), usize>,
    F: FnOnce(EnvStructure<SingletonSpace, IndexSpace>) -> A,
{
    let mut env = DeterministicBandit::from_values(vec![0.0, 1.0]).as_stateful(0);
    let mut agent = make_agent(env.structure());

    let num_eval_steps = 1000;
    let num_steps = num_train_steps + num_eval_steps;
    let mut step_count = 0;
    let mut action_1_count = 0;

    simulation::run(&mut env, &mut agent, |step| {
        step_count += 1;
        if step_count < num_train_steps {
            // Training
            false
        } else if step_count < num_steps {
            // Evaluation
            action_1_count += (step.action == 1) as u64;
            false
        } else {
            // Done
            true
        }
    });

    assert!(action_1_count >= ((num_eval_steps as f64) * threshold) as u64);
}
