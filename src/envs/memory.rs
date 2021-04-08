use super::{EnvStructure, Environment};
use crate::spaces::{IndexSpace, Space};
use rand::prelude::*;

/// Memory Game Environment
///
/// * The environment consists of (N + M) states.
/// * An episode starts in a state [0, N) uniformly at random.
/// * Step i (in [0, M)) transitions to state N + i with 0 reward regardless of the action.
/// * On step M, the agent chooses 1 of N actions
///     and if the action index matches the index of the inital state
///     then the agent earns +1 reward, otherwise it earns -1 reward.
///     This step is terminal.
/// * Every episode has length M + 1.
#[derive(Debug, Clone)]
pub struct MemoryGame {
    /// The number of actions; `N`.
    pub num_actions: usize,
    /// Length of remembered history required to solve the environment; `M`.
    pub history_len: usize,
}

impl Default for MemoryGame {
    fn default() -> Self {
        Self {
            num_actions: 2,
            history_len: 1,
        }
    }
}

impl MemoryGame {
    /// Create a new MemoryGame instance
    ///
    /// # Args
    /// * `num_actions` - Number of possible actions (N).
    /// * `history_len` - Length of remembered history required to solve the environment (M).
    pub fn new(num_actions: usize, history_len: usize) -> Self {
        Self {
            num_actions,
            history_len,
        }
    }
}

impl Environment for MemoryGame {
    /// (current_state, initial_state)
    type State = (usize, usize);
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        let state = rng.gen_range(0, self.num_actions);
        (state, state)
    }

    fn observe(
        &self,
        state: &Self::State,
        _rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        state.0
    }

    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        _rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        let (current_state, initial_state) = state;
        if current_state == self.num_actions + self.history_len - 1 {
            let reward = if *action == initial_state { 1.0 } else { -1.0 };
            (None, reward, true)
        } else {
            let new_state = if current_state < self.num_actions {
                self.num_actions
            } else {
                current_state + 1
            };
            (Some((new_state, initial_state)), 0.0, false)
        }
    }

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        EnvStructure {
            observation_space: IndexSpace::new(self.num_actions + self.history_len),
            action_space: IndexSpace::new(self.num_actions),
            reward_range: (-1.0, 1.0),
            discount_factor: 1.0,
        }
    }
}

#[cfg(test)]
mod memory_game {
    use super::super::testing;
    use super::*;

    #[test]
    fn run_default() {
        testing::run_stateless(MemoryGame::default(), 1000, 0);
    }
}
