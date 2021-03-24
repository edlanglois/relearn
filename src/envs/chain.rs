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
    fn swap(self) -> Self {
        match self {
            Move::Left => Move::Right,
            Move::Right => Move::Left,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{AsStateful, StatefulEnvironment};
    use super::*;
    use crate::agents::RandomAgent;
    use crate::simulator;

    #[test]
    fn run_chain_default() {
        let mut env = Chain::new(None, None).as_stateful(4);
        let mut agent = RandomAgent::new(env.structure().action_space, 5);
        let mut step_count: u32 = 0;
        simulator::run(&mut env, &mut agent, &mut |_| {
            step_count += 1;
            step_count < 1000
        });
    }
}
