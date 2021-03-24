//! Environment traits
use crate::spaces::Space;
use rand::rngs::StdRng;
use std::f32;

/// A reinforcement learning environment.
///
/// This defines the environment dynamics and strucutre.
/// It does not internally manage state.
///
/// Every environment implements AsStatefulEnvironment
/// so that it can be converted to a StatefulEnvironment.
pub trait Environment {
    type State;
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Sample a new initial state.
    fn initial_state(&self, rng: &mut StdRng) -> Self::State;

    /// Sample an observation for a state.
    fn observe(
        &self,
        state: &Self::State,
        rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element;

    /// Sample a state transition.
    ///
    /// # Returns
    /// * `state`: The resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition
    /// * `episode_done`: Whether this step ends the episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f32, bool);

    /// The structure of this environment.
    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace>;
}

/// A reinforcement learning environment with internal state.
pub trait StatefulEnvironment {
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Take a step in the environment.
    ///
    /// This may panic if the state has not be initialized with reset()
    /// after initialization or after a step returned `episod_done = True`.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition
    /// * `episode_done`: Whether this step ends the episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    // TODO: Why not make reset() optional and have new() / step() self-reset?
    fn step(
        &mut self,
        action: &<Self::ActionSpace as Space>::Element,
    ) -> (
        Option<<Self::ObservationSpace as Space>::Element>,
        f32,
        bool,
    );

    /// Reset the environment to an initial state.
    ///
    /// Must be called before each new episode.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    fn reset(&mut self) -> <Self::ObservationSpace as Space>::Element;

    /// Get the structure of this environment.
    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace>;
}

/// The external structure of an environment.
#[derive(Debug)]
pub struct EnvStructure<OS: Space, AS: Space> {
    /// Space containing all possible observations.
    ///
    /// This is not required to be tight:
    /// this space may contain elements that can never be produced as a state observation.
    pub observation_space: OS,
    /// The space of possible actions.
    ///
    /// Every element in this space must be a valid action.
    pub action_space: AS,
    /// A lower and upper bound on possible reward values.
    ///
    /// These bounds are not required to be tight but ideally will be as tight as possible.
    pub reward_range: (f32, f32),
    /// A discount factor applied to future rewards.
    pub discount_factor: f32,
}

#[cfg(test)]
pub mod env_checks {
    use super::super::as_stateful::AsStateful;
    use super::*;
    use crate::agents::RandomAgent;
    use crate::simulator;

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
}
