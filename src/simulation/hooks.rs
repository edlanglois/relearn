//! Simulator hooks.
use crate::agents::Step;
use crate::envs::EnvStructure;
use crate::logging::{Event, Loggable, LoggerHelper, TimeSeriesLogger};
use crate::spaces::{ElementRefInto, FiniteSpace, Space};
use impl_trait_for_tuples::impl_for_tuples;

/// Build a [`SimulationHook`] for a given environment structure.
pub trait BuildStructuredHook<OS: Space, AS: Space> {
    type Hook: SimulationHook<OS::Element, AS::Element>;

    /// Build a simulation hook.
    ///
    /// # Args
    /// * `env` - Environment structure.
    /// * `num_threads` - Total number of worker threads for the simulation.
    /// * `thread_index` - An index in `[0, num_threads)` uniquely identifying
    ///                    the simulation thread in which this hook will run.
    fn build_hook(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        num_threads: usize,
        thread_index: usize,
    ) -> Self::Hook;
}

/// A simulation hook.
///
/// A callback function called on each step.
pub trait SimulationHook<O, A> {
    /// Call the hook at the start of the simulation.
    ///
    /// # Returns
    /// Whether the simulation should run.
    fn start(&mut self, _logger: &mut dyn TimeSeriesLogger) -> bool {
        true
    }

    /// Call the hook on the current step.
    ///
    /// # Args
    /// * `step` - The most recent environment step.
    /// * `logger` - A logger.
    ///
    /// # Returns
    /// Whether the simulation should continue after this step.
    fn call(&mut self, step: &Step<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool;
}

/// A generic simulation hook that applies to every state, action, and logger.
pub trait GenericSimulationHook {
    /// Call the hook at the start of the simulation.
    ///
    /// # Returns
    /// Whether the simulation should run.
    fn start(&mut self, _logger: &mut dyn TimeSeriesLogger) -> bool {
        true
    }

    /// Call the hook on the current step.
    ///
    /// # Args
    /// * `step` - The most recent environment step.
    /// * `logger` - A logger.
    ///
    /// # Returns
    /// Whether the simulation should continue after this step.
    fn call<O, A>(&mut self, step: &Step<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool;
}

impl<O, A, T: GenericSimulationHook> SimulationHook<O, A> for T {
    fn start(&mut self, logger: &mut dyn TimeSeriesLogger) -> bool {
        GenericSimulationHook::start(self, logger)
    }

    fn call(&mut self, step: &Step<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool {
        GenericSimulationHook::call(self, step, logger)
    }
}

/// Divide `total` almost evently into `parts` that sum up to `total`.
///
/// Each part gets `ceil(total / parts)` or `floor(total / parts)` depending on `part_index`.
const fn partition_div(total: u64, parts: u64, part_index: u64) -> u64 {
    let remainder = if total % parts > part_index { 1 } else { 0 };
    total / parts + remainder
}

/// Configuration for [`StepLimit`].
///
/// Sets a maximum total number of episodes across all simulation threads.
/// Each thread runs for a maximum of `max_episodes / num_threads` steps (floor or ceil).
/// The total number of episodes cannot be less than the number of threads because a hook cannot
/// stop the simulation before the first step.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StepLimitConfig {
    pub max_steps: u64,
}

impl StepLimitConfig {
    pub const fn new(max_steps: u64) -> Self {
        Self { max_steps }
    }
}

impl<OS: Space, AS: Space> BuildStructuredHook<OS, AS> for StepLimitConfig {
    type Hook = StepLimit;

    fn build_hook(
        &self,
        _env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        num_threads: usize,
        thread_index: usize,
    ) -> Self::Hook {
        StepLimit::new(partition_div(
            self.max_steps,
            num_threads as u64,
            thread_index as u64,
        ))
    }
}

/// A hook that stops the simulation after a maximum number of steps.
#[derive(Debug, Clone, Copy)]
pub struct StepLimit {
    steps_remaining: u64,
}

impl StepLimit {
    /// Create a new `StepLimit` hook.
    pub const fn new(max_steps: u64) -> Self {
        Self {
            steps_remaining: max_steps,
        }
    }
}

impl GenericSimulationHook for StepLimit {
    fn start(&mut self, _: &mut dyn TimeSeriesLogger) -> bool {
        self.steps_remaining > 0
    }
    fn call<O, A>(&mut self, _: &Step<O, A>, _: &mut dyn TimeSeriesLogger) -> bool {
        self.steps_remaining -= 1;
        self.steps_remaining > 0
    }
}

/// Configuration for [`EpisodeLimit`].
///
/// Sets a maximum total number of episode across all simulation threads.
/// Each thread runs for a maximum of `max_steps / num_threads` steps (floor or ceil).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct EpisodeLimitConfig {
    pub max_episodes: u64,
}

impl EpisodeLimitConfig {
    pub const fn new(max_episodes: u64) -> Self {
        Self { max_episodes }
    }
}

impl<OS: Space, AS: Space> BuildStructuredHook<OS, AS> for EpisodeLimitConfig {
    type Hook = EpisodeLimit;

    fn build_hook(
        &self,
        _env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        num_threads: usize,
        thread_index: usize,
    ) -> Self::Hook {
        EpisodeLimit::new(partition_div(
            self.max_episodes,
            num_threads as u64,
            thread_index as u64,
        ))
    }
}

/// A hook that stops the simulation after a maximum number of episodes.
#[derive(Debug, Clone, Copy)]
pub struct EpisodeLimit {
    episodes_remaining: u64,
}

impl EpisodeLimit {
    /// Create a new `EpisodeLimit` hook.
    pub const fn new(max_episodes: u64) -> Self {
        Self {
            episodes_remaining: max_episodes,
        }
    }
}

impl GenericSimulationHook for EpisodeLimit {
    fn start(&mut self, _: &mut dyn TimeSeriesLogger) -> bool {
        self.episodes_remaining > 0
    }

    fn call<O, A>(&mut self, step: &Step<O, A>, _: &mut dyn TimeSeriesLogger) -> bool {
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
    fn call(&mut self, step: &Step<O, A>, _: &mut dyn TimeSeriesLogger) -> bool {
        (self.f)(step)
    }
}

/// Configuration for [`StepLogger`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StepLoggerConfig;

impl<OS, AS> BuildStructuredHook<OS, AS> for StepLoggerConfig
where
    OS: ElementRefInto<Loggable>,
    AS: ElementRefInto<Loggable>,
{
    type Hook = StepLogger<OS, AS>;

    fn build_hook(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _num_threads: usize,
        _thread_index: usize,
    ) -> Self::Hook {
        StepLogger::new(env.observation_space(), env.action_space())
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
    fn call(
        &mut self,
        step: &Step<OS::Element, AS::Element>,
        logger: &mut dyn TimeSeriesLogger,
    ) -> bool {
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

// For a collection (list or tuple) of hooks, stop if any hook requests a stop.

impl GenericSimulationHook for () {
    fn call<O, A>(&mut self, _: &Step<O, A>, _: &mut dyn TimeSeriesLogger) -> bool {
        true
    }
}

#[impl_for_tuples(1, 12)]
impl<O, A> SimulationHook<O, A> for Tuple {
    fn start(&mut self, logger: &mut dyn TimeSeriesLogger) -> bool {
        for_tuples!( #( self.Tuple.start(logger) )&* )
    }

    fn call(&mut self, step: &Step<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool {
        for_tuples!( #( self.Tuple.call(step, logger) )&* )
    }
}

impl<T, O, A> SimulationHook<O, A> for [T]
where
    T: SimulationHook<O, A>,
{
    fn start(&mut self, logger: &mut dyn TimeSeriesLogger) -> bool {
        self.iter_mut().all(|h| h.start(logger))
    }

    fn call(&mut self, step: &Step<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool {
        self.iter_mut().all(|h| h.call(step, logger))
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
    fn call(&mut self, step: &Step<O, AS::Element>, _: &mut dyn TimeSeriesLogger) -> bool {
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
    fn call<O, A>(&mut self, step: &Step<O, A>, _logger: &mut dyn TimeSeriesLogger) -> bool {
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
