//! Chain environment
use super::{EnvStructure, Environment};
use crate::spaces::{IndexSpace, IndexedTypeSpace, Space};
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
#[derive(Debug, Clone)]
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

impl Environment for Chain {
    type State = u64;
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexedTypeSpace<Move>;

    fn initial_state(&self, _rng: &mut StdRng) -> Self::State {
        0
    }

    fn observe(
        &self,
        state: &Self::State,
        _rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        (*state).try_into().unwrap()
    }

    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
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

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        EnvStructure {
            observation_space: IndexSpace::new(self.size.try_into().unwrap()),
            action_space: Self::ActionSpace::new(),
            reward_range: (0.0, 10.0),
            discount_factor: self.discount_factor,
        }
    }
}

#[derive(Debug, Copy, Clone, Indexed)]
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
#[allow(clippy::module_inception)]
mod chain {
    use super::super::testing;
    use super::*;

    #[test]
    fn run_default() {
        testing::run_stateless(Chain::default(), 1000, 0);
    }
}
