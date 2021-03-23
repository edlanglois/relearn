use super::{EnvStructure, Environment};
use crate::logging::Loggable;
use crate::spaces::{FiniteSpace, IndexSpace, Space};
use rand::prelude::*;
use std::convert::TryInto;
use std::fmt;

/// Chain Environment
///
/// Consists of n states in a line with 2 actions.
/// * Action 0 moves back to the start for 2 reward.
/// * Action 1 moves forward for 0 reward in all states but the last.
///     In the last state, taking action 1 is a self-transition with 10 reward.
/// * Every action has a 0.2 chance of "slipping" and taking the opposite action.
///
/// Described in "Bayesian Q-learning" by Dearden, Friedman and Russel (1998)
#[derive(Debug)]
pub struct Chain {
    size: u32,
    discount_factor: f32,
}

impl Chain {
    pub fn new(size: Option<u32>, discount_factor: Option<f32>) -> Self {
        Self {
            size: size.unwrap_or(5),
            discount_factor: discount_factor.unwrap_or(0.95),
        }
    }
}

impl Environment for Chain {
    type State = u32;
    type ObservationSpace = IndexSpace;
    type ActionSpace = MoveSpace;

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
    ) -> (Option<Self::State>, f32, bool) {
        let mut action = *action;
        if rng.gen::<f32>() < 0.2 {
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
            action_space: MoveSpace {},
            reward_range: (0.0, 10.0),
            discount_factor: self.discount_factor,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Move {
    Left,
    Right,
}

impl Move {
    fn swap(self) -> Self {
        match self {
            Move::Left => Move::Right,
            Move::Right => Move::Left,
        }
    }
}

// TODO: Automate the following. Macro or template impl
#[derive(Debug)]
pub struct MoveSpace {}

impl fmt::Display for MoveSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MoveSpace")
    }
}

impl Distribution<Move> for MoveSpace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Move {
        self.from_index(rng.gen_range(0..self.size())).unwrap()
    }
}

impl Space for MoveSpace {
    type Element = Move;

    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }

    fn as_loggable(&self, value: &Self::Element) -> Loggable {
        Loggable::IndexSample {
            value: self.to_index(value),
            size: self.size(),
        }
    }
}

impl FiniteSpace for MoveSpace {
    fn size(&self) -> usize {
        2
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        match element {
            Move::Left => 0,
            Move::Right => 1,
        }
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        match index {
            0 => Some(Move::Left),
            1 => Some(Move::Right),
            _ => None,
        }
    }
}
