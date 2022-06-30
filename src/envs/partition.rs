use super::{CloneBuild, EnvStructure, Environment, Successor};
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::spaces::{
    BooleanSpace, Indexed, IndexedTypeSpace, IntervalSpace, OptionSpace, PowerSpace, TupleSpace2,
};
use crate::Prng;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const NUM_FEATURES: usize = 10;
/// Elements to partition
pub type Element = [bool; NUM_FEATURES];

/// Environment where the goal is to partition vectors based on supervision
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionGame;

impl CloneBuild for PartitionGame {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Supervisor {
    AxisAligned(usize),
}

impl Supervisor {
    const fn classify(self, element: &Element) -> Classification {
        match self {
            Self::AxisAligned(axis) => {
                if element[axis] {
                    Classification::Right
                } else {
                    Classification::Left
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Indexed)]
pub enum Classification {
    Left,
    Right,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Indexed)]
pub enum Action {
    ClassifyLeft,
    ClassifyRight,
    // Query,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PartitionGameState {
    supervisor: Supervisor,
    element: Element,
    /// (prev_element, classification)
    feedback: Option<(Element, Classification)>,
}

pub type ElementSpace = PowerSpace<BooleanSpace, NUM_FEATURES>;
pub type FeedbackSpace = TupleSpace2<ElementSpace, IndexedTypeSpace<Classification>>;

impl EnvStructure for PartitionGame {
    type ObservationSpace = TupleSpace2<ElementSpace, OptionSpace<FeedbackSpace>>;
    type ActionSpace = IndexedTypeSpace<Action>;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        Default::default()
    }

    fn action_space(&self) -> Self::ActionSpace {
        Default::default()
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        IntervalSpace::new(Reward(-1.0), Reward(1.0))
    }

    fn discount_factor(&self) -> f64 {
        0.999
    }
}

impl Environment for PartitionGame {
    type State = PartitionGameState;
    /// Current element and feedback
    type Observation = (Element, Option<(Element, Classification)>);
    type Action = Action;
    type Feedback = Reward;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        let supervisor = Supervisor::AxisAligned(rng.gen_range(0..NUM_FEATURES));
        let element = rng.gen();
        PartitionGameState {
            supervisor,
            element,
            feedback: None,
        }
    }

    fn observe(&self, state: &Self::State, _: &mut Prng) -> Self::Observation {
        (state.element, state.feedback)
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        _: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        let label = state.supervisor.classify(&state.element);
        let reward = match (label, action) {
            (Classification::Left, Action::ClassifyLeft)
            | (Classification::Right, Action::ClassifyRight) => 1.0,
            _ => -1.0,
        };
        (
            Successor::Continue(PartitionGameState {
                supervisor: state.supervisor,
                element: rng.gen(),
                // Feedback on the previous state and action
                feedback: Some((state.element, label)),
            }),
            Reward(reward),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing;
    use super::*;

    #[test]
    fn run_default() {
        testing::check_structured_env(&PartitionGame::default(), 1000, 0);
    }
}
