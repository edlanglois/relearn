//! Chain environment
use super::{BuildEnvError, BuildEnv, EnvStructure, Environment};
use crate::spaces::{IndexSpace, IndexedTypeSpace};
use rand::prelude::*;
use rust_rl_derive::Indexed;
use std::convert::TryInto;

/// Chain Environment
///
/// Consists of n states in a line with 2 actions.
/// * Action 0 moves back to the start for 2 reward.
/// * Action 1 moves forward for 0 reward in all states but the last.
///     In the last state, taking action 1 is a self-transition with 10 reward.
/// * Every action has a 0.2 chance of "slipping" and taking the opposite action.
///
/// Described in "Bayesian Q-learning" by Dearden, Friedman and Russel (1998)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chain {
    pub size: u64,
    pub discount_factor: f64,
}

impl Chain {
    pub const fn new(size: u64, discount_factor: f64) -> Self {
        Self {
            size,
            discount_factor,
        }
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self {
            size: 5,
            discount_factor: 0.95,
        }
    }
}

/// Chain is its own configuration
impl BuildEnv<Self> for Chain {
    fn build_env(&self, _seed: u64) -> Result<Self, BuildEnvError> {
        Ok(*self)
    }
}

impl EnvStructure for Chain {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexedTypeSpace<Move>;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(self.size.try_into().unwrap())
    }

    fn action_space(&self) -> Self::ActionSpace {
        Self::ActionSpace::new()
    }

    fn reward_range(&self) -> (f64, f64) {
        (0.0, 10.0)
    }

    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl Environment for Chain {
    type State = u64;
    type Observation = usize;
    type Action = Move;

    fn initial_state(&self, _rng: &mut StdRng) -> Self::State {
        0
    }

    fn observe(&self, state: &Self::State, _rng: &mut StdRng) -> Self::Observation {
        (*state).try_into().unwrap()
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        let mut action = *action;
        if rng.gen::<f64>() < 0.2 {
            action = action.swap();
        }
        let (state, reward) = match action {
            Move::Left => (0, 2.0),
            Move::Right => {
                if state == self.size - 1 {
                    (state, 10.0)
                } else {
                    (state + 1, 0.0)
                }
            }
        };
        (Some(state), reward, false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Indexed)]
pub enum Move {
    Left,
    Right,
}

impl Move {
    const fn swap(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing;
    use super::*;

    #[test]
    fn run_default() {
        testing::run_stateless(Chain::default(), 1000, 0);
    }
}
