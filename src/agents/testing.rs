//! Agent testing utilities
use crate::envs::{AsStateful, DeterministicBandit, StatefulEnvironment};
use crate::simulation;
use crate::simulation::hooks::{IndexedActionCounter, StepLimit};
use crate::spaces::{IndexSpace, SingletonSpace};
use crate::{Actor, Agent, EnvStructure};

/// Check that the agent can be trained to perform well on a trivial bandit environment.
///
/// The environment is a deterministic multi-armed bandit with two arms:
/// the first arm always gives 0 reward and the second 1.
pub fn train_deterministic_bandit<A, F>(make_agent: F, num_train_steps: u64, threshold: f64)
where
    A: Agent<(), usize> + Actor<(), usize>,
    F: FnOnce(EnvStructure<SingletonSpace, IndexSpace>) -> A,
{
    let mut env = DeterministicBandit::from_values(vec![0.0, 1.0]).as_stateful(0);
    let env_structure = env.structure();
    let action_space = env_structure.action_space.clone();
    let mut agent = make_agent(env_structure);

    // Training
    if num_train_steps > 0 {
        simulation::run_agent(
            &mut env,
            &mut agent,
            &mut (),
            &mut StepLimit::new(num_train_steps),
        );
    }

    // Evaluation
    let num_eval_steps = 1000;

    let action_counter = IndexedActionCounter::new(action_space);
    let mut hooks = (action_counter, StepLimit::new(num_eval_steps));
    simulation::run_actor(&mut env, &mut agent, &mut (), &mut hooks);

    let action_1_count = hooks.0.counts[1];
    assert!(action_1_count >= ((num_eval_steps as f64) * threshold) as u64);
}
