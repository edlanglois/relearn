//! Chain environment
use super::{CloneBuild, EnvStructure, Environment, Successor};
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::spaces::{IndexSpace, IndexedTypeSpace, IntervalSpace};
use crate::Prng;
use rand::prelude::*;
use relearn_derive::Indexed;
use serde::{Deserialize, Serialize};

/// Chain Environment
///
/// Consists of n states in a line with 2 actions.
/// * Action 0 moves back to the start for 2 reward.
/// * Action 1 moves forward for 0 reward in all states but the last.
///     In the last state, taking action 1 is a self-transition with 10 reward.
/// * Every action has a 0.2 chance of "slipping" and taking the opposite action.
///
/// Described in "Bayesian Q-learning" by Dearden, Friedman and Russel (1998)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Chain {
    pub size: usize,
    pub discount_factor: f64,
}

impl CloneBuild for Chain {}

impl Chain {
    #[must_use]
    pub const fn new(size: usize, discount_factor: f64) -> Self {
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

impl EnvStructure for Chain {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexedTypeSpace<Move>;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(self.size)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Self::ActionSpace::new()
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        IntervalSpace::new(Reward(0.0), Reward(10.0))
    }

    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl Environment for Chain {
    type State = usize;
    type Observation = usize;
    type Action = Move;
    type Feedback = Reward;

    fn initial_state(&self, _: &mut Prng) -> Self::State {
        0
    }

    fn observe(&self, state: &Self::State, _: &mut Prng) -> Self::Observation {
        *state
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        _: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        let mut action = *action;
        if rng.gen::<f32>() < 0.2 {
            action = action.invert();
        }
        let (next_state, reward) = match action {
            Move::Left => (0, 2.0),
            Move::Right => {
                if state == self.size - 1 {
                    (state, 10.0)
                } else {
                    (state + 1, 0.0)
                }
            }
        };
        (Successor::Continue(next_state), reward.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Indexed)]
pub enum Move {
    Left,
    Right,
}

impl Move {
    const fn invert(self) -> Self {
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
        testing::check_structured_env(&Chain::default(), 1000, 0);
    }
}
