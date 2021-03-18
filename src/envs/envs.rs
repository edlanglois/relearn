use crate::spaces::Space;
use rand::{rngs::StdRng, SeedableRng};
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

/// Creates a StatefulEnvironment out of an Environment
pub struct EnvWithState<E: Environment> {
    pub env: E,
    state: Option<E::State>,
    rng: StdRng,
}

impl<E: Environment> EnvWithState<E> {
    pub fn new(env: E, seed: u64) -> Self {
        Self {
            env,
            state: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<E: Environment> StatefulEnvironment for EnvWithState<E> {
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;

    fn step(
        &mut self,
        action: &<Self::ActionSpace as Space>::Element,
    ) -> (
        Option<<Self::ObservationSpace as Space>::Element>,
        f32,
        bool,
    ) {
        let state = self
            .state
            .take()
            .expect("Must call reset() before the start of each episode");
        let (next_state, reward, episode_done) = self.env.step(state, action, &mut self.rng);
        self.state = next_state;
        let observation = match self.state.as_ref() {
            Some(s) => Some(self.env.observe(s, &mut self.rng)),
            None => None,
        };
        (observation, reward, episode_done)
    }

    fn reset(&mut self) -> <Self::ObservationSpace as Space>::Element {
        let state = self.env.initial_state(&mut self.rng);
        let observation = self.env.observe(&state, &mut self.rng);
        self.state = Some(state);
        observation
    }

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        self.env.structure()
    }
}

/// Supports conversion to a stateful environment
pub trait AsStateful {
    type Output: StatefulEnvironment;

    /// Convert into a stateful environment.
    fn as_stateful(self, seed: u64) -> Self::Output;
}

impl<E: Environment> AsStateful for E {
    type Output = EnvWithState<E>;

    fn as_stateful(self, seed: u64) -> Self::Output {
        Self::Output::new(self, seed)
    }
}
