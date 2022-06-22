///! Parameter schedules --- functions of the global step count during training.
use crate::agents::{buffers::HistoryDataBound, ActorMode};
use serde::{Deserialize, Serialize};

/// Selects the exploration rate as a function the elapsed step count.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExplorationRateSchedule {
    Constant(f64),
    LinearAnnealed {
        start: f64,
        end: f64,
        /// Number of steps to reach the `end` value.
        period: u64,
    },
}

/// Default initialization based on the [Rainbow DQN][rdqn] paper.
///
/// Most appropriate for the Atari game environments.
/// Consider experimenting for other environments.
///
/// [rdqn]: https://arxiv.org/pdf/1710.02298.pdf
impl Default for ExplorationRateSchedule {
    fn default() -> Self {
        Self::LinearAnnealed {
            start: 1.0,
            end: 0.1,
            period: 10_000_000,
        }
    }
}

impl ExplorationRateSchedule {
    #[must_use]
    pub fn exploration_rate(&self, global_steps: u64, mode: ActorMode) -> f64 {
        use ExplorationRateSchedule::{Constant, LinearAnnealed};
        match (mode, self) {
            (ActorMode::Evaluation, _) => 0.0,
            (ActorMode::Training, Constant(rate)) => *rate,
            (ActorMode::Training, LinearAnnealed { start, end, period }) => {
                (global_steps as f64 / *period as f64).min(1.0) * (end - start) + start
            }
        }
    }
}

// TODO: Would be interesting to try and dynamically adjust the collections so that it takes
// a fixed amount of time on each update.
/// Selects the amount of data to collect on each update.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataCollectionSchedule {
    /// Collect the same amount of data on each step.
    Constant(usize),
    /// Collect a minimum amount of data at first, then another amount on every subsequent step.
    FirstRest { first: usize, rest: usize },
}

impl DataCollectionSchedule {
    #[must_use]
    pub fn update_size(&self, global_steps: u64) -> HistoryDataBound {
        use DataCollectionSchedule::*;
        let min_steps = match self {
            Constant(value) => *value,
            FirstRest { first, rest: _ } if global_steps < *first as u64 => *first,
            FirstRest { first: _, rest } => *rest,
        };
        HistoryDataBound::with_default_slack(min_steps)
    }
}
