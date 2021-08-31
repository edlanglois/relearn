//! Simulator hooks.
use crate::agents::Step;
use crate::logging::{Event, Loggable, LoggerHelper, TimeSeriesLogger};
use crate::spaces::{ElementRefInto, FiniteSpace};
use impl_trait_for_tuples::impl_for_tuples;

/// A simulation hook.
///
/// A callback function called on each step.
pub trait SimulationHook<O, A> {
    /// Call the hook on the current step.
    ///
    /// # Args
    /// * `step` - The most recent environment step.
    /// * `logger` - A logger.
    ///
    /// # Returns
    /// Whether the simulation should continue after this step.
    fn call<L>(&mut self, step: &Step<O, A>, logger: &mut L) -> bool
    where
        L: TimeSeriesLogger + ?Sized;
}

/// A generic simulation hook that applies to every state, action, and logger.
pub trait GenericSimulationHook {
    /// Call the hook on the current step.
    ///
    /// # Args
    /// * `step` - The most recent environment step.
    /// * `logger` - A logger.
    ///
    /// # Returns
    /// Whether the simulation should continue after this step.
    fn call<O, A, L: TimeSeriesLogger + ?Sized>(
        &mut self,
        step: &Step<O, A>,
        logger: &mut L,
    ) -> bool;
}

impl<O, A, T: GenericSimulationHook> SimulationHook<O, A> for T {
    fn call<L>(&mut self, step: &Step<O, A>, logger: &mut L) -> bool
    where
        L: TimeSeriesLogger + ?Sized,
    {
        GenericSimulationHook::call(self, step, logger)
    }
}

/// A hook that stops the simulation after a maximum number of steps.
#[derive(Debug, Clone, Copy)]
pub struct StepLimit {
    steps_remaining: u64,
}

impl StepLimit {
    /// Create a new `StepLimit` hook.
    ///
    /// `max_steps` must be >= 1 because hooks cannot stop the simulation before the first step.
    pub fn new(max_steps: u64) -> Self {
        assert!(max_steps > 0);
        Self {
            steps_remaining: max_steps,
        }
    }
}

impl GenericSimulationHook for StepLimit {
    fn call<O, A, L: TimeSeriesLogger + ?Sized>(&mut self, _: &Step<O, A>, _: &mut L) -> bool {
        self.steps_remaining -= 1;
        self.steps_remaining > 0
    }
}

/// A hook that stops the simulation after a maximum number of episodes.
#[derive(Debug, Clone, Copy)]
pub struct EpisodeLimit {
    episodes_remaining: u64,
}

impl EpisodeLimit {
    /// Create a new `EpisodeLimit` hook.
    ///
    /// `max_episodes` must be `>= 1` because hooks cannot stop the simulation before the first
    /// step and this hook only stops at the end of an episode.
    pub fn new(max_episodes: u64) -> Self {
        assert!(max_episodes > 0);
        Self {
            episodes_remaining: max_episodes,
        }
    }
}

impl GenericSimulationHook for EpisodeLimit {
    fn call<O, A, L: TimeSeriesLogger + ?Sized>(&mut self, step: &Step<O, A>, _: &mut L) -> bool {
        if step.episode_done {
            self.episodes_remaining -= 1;
            self.episodes_remaining > 0
        } else {
            true
        }
    }
}

/// A simulation hook defined from a closure.
#[derive(Debug, Clone, Copy)]
pub struct ClosureHook<F> {
    f: F,
}

impl<F> From<F> for ClosureHook<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

impl<O, A, F> SimulationHook<O, A> for ClosureHook<F>
where
    F: FnMut(&Step<O, A>) -> bool,
{
    fn call<L: TimeSeriesLogger + ?Sized>(&mut self, step: &Step<O, A>, _: &mut L) -> bool {
        (self.f)(step)
    }
}

/// A hook that logs step and episode statistics.
#[derive(Debug, Clone, Copy)]
pub struct StepLogger<OS, AS> {
    pub observation_space: OS,
    pub action_space: AS,

    /// Length of the current episode
    episode_length: u64,
    episode_reward: f64,
}

impl<OS, AS> StepLogger<OS, AS> {
    pub const fn new(observation_space: OS, action_space: AS) -> Self {
        Self {
            observation_space,
            action_space,
            episode_length: 0,
            episode_reward: 0.0,
        }
    }
}

impl<OS, AS> SimulationHook<OS::Element, AS::Element> for StepLogger<OS, AS>
where
    OS: ElementRefInto<Loggable>,
    AS: ElementRefInto<Loggable>,
{
    fn call<L>(&mut self, step: &Step<OS::Element, AS::Element>, logger: &mut L) -> bool
    where
        L: TimeSeriesLogger + ?Sized,
    {
        let mut step_logger = logger.event_logger(Event::EnvStep);
        step_logger.unwrap_log_scalar("reward", step.reward);
        step_logger.unwrap_log(
            "observation",
            self.observation_space.elem_ref_into(&step.observation),
        );
        step_logger.unwrap_log("action", self.action_space.elem_ref_into(&step.action));
        logger.end_event(Event::EnvStep).unwrap();

        self.episode_length += 1;
        self.episode_reward += step.reward;
        if step.episode_done {
            let mut episode_logger = logger.event_logger(Event::EnvEpisode);
            episode_logger.unwrap_log("length", self.episode_length as f64);
            self.episode_length = 0;
            episode_logger.unwrap_log("reward", self.episode_reward);
            self.episode_reward = 0.0;
            logger.end_event(Event::EnvEpisode).unwrap();
        }

        true
    }
}

// TODO: Generate for other tuple sizes with a macro
// Consider impl_trait_for_tuples crate

// For a tuple of hooks, continue if all allow continuing.

impl GenericSimulationHook for () {
    fn call<O, A, L: TimeSeriesLogger + ?Sized>(&mut self, _: &Step<O, A>, _: &mut L) -> bool {
        true
    }
}

#[impl_for_tuples(1, 12)]
impl<O, A> SimulationHook<O, A> for Tuple {
    fn call<L>(&mut self, step: &Step<O, A>, logger: &mut L) -> bool
    where
        L: TimeSeriesLogger + ?Sized,
    {
        for_tuples!( #( self.Tuple.call(step, logger) )&* )
    }
}

/// Count occurrences of each action by index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexedActionCounter<AS> {
    pub action_space: AS,
    pub counts: Vec<u64>,
}
impl<AS: FiniteSpace> IndexedActionCounter<AS> {
    pub fn new(action_space: AS) -> Self {
        let num_actions = action_space.size();
        Self {
            action_space,
            counts: vec![0; num_actions],
        }
    }
}
impl<O, AS: FiniteSpace> SimulationHook<O, AS::Element> for IndexedActionCounter<AS> {
    fn call<L>(&mut self, step: &Step<O, AS::Element>, _: &mut L) -> bool
    where
        L: TimeSeriesLogger + ?Sized,
    {
        self.counts[self.action_space.to_index(&step.action)] += 1;
        true
    }
}

/// Collect reward statistics.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RewardStatistics {
    /// Total reward for completed episodes
    total_episode_reward: f64,
    /// Reward for the current incomplete episode
    partial_reward: f64,
    /// Number of completed steps
    num_steps: u64,
    /// Number of completed episodes
    num_episodes: u64,
}

impl RewardStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mean reward per step.
    pub fn mean_step_reward(&self) -> f64 {
        (self.total_episode_reward + self.partial_reward) / (self.num_steps as f64)
    }

    /// Mean reward per episode.
    pub fn mean_episode_reward(&self) -> f64 {
        self.total_episode_reward / (self.num_episodes as f64)
    }
}

impl GenericSimulationHook for RewardStatistics {
    fn call<O, A, L: TimeSeriesLogger + ?Sized>(
        &mut self,
        step: &Step<O, A>,
        _logger: &mut L,
    ) -> bool {
        self.partial_reward += step.reward;
        self.num_steps += 1;
        if step.episode_done {
            self.total_episode_reward += self.partial_reward;
            self.partial_reward = 0.0;
            self.num_episodes += 1;
        }
        true
    }
}
