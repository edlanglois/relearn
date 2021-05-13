//! Simulator hooks.
use crate::agents::Step;
use crate::logging::{Event, Loggable, Logger};
use crate::spaces::{ElementRefInto, FiniteSpace};

/// A simulation hook.
///
/// A callback function called on each step.
pub trait SimulationHook<O, A, L: Logger + ?Sized> {
    /// Call the hook on the current step.
    ///
    /// # Args
    /// * `step` - The most recent environment step.
    /// * `logger` - A logger.
    ///
    /// # Returns
    /// Whether the simulation should continue or stop after this step.
    fn call(&mut self, step: &Step<O, A>, logger: &mut L) -> bool;
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
    /// Whether the simulation should continue or stop after this step.
    fn call<O, A, L: Logger + ?Sized>(&mut self, step: &Step<O, A>, logger: &mut L) -> bool;
}

impl<O, A, L: Logger + ?Sized, T: GenericSimulationHook> SimulationHook<O, A, L> for T {
    fn call(&mut self, step: &Step<O, A>, logger: &mut L) -> bool {
        GenericSimulationHook::call(self, step, logger)
    }
}

/// A hook that stops the simulation after a maximum number of steps.
#[derive(Debug)]
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
    fn call<O, A, L: Logger + ?Sized>(&mut self, _: &Step<O, A>, _: &mut L) -> bool {
        self.steps_remaining -= 1;
        self.steps_remaining > 0
    }
}

/// A hook that stops the simulation after a maximum number of episodes.
#[derive(Debug)]
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
    fn call<O, A, L: Logger + ?Sized>(&mut self, step: &Step<O, A>, _: &mut L) -> bool {
        if step.episode_done {
            self.episodes_remaining -= 1;
            self.episodes_remaining > 0
        } else {
            true
        }
    }
}

/// A simulation hook defined from a closure.
pub struct ClosureHook<F> {
    f: F,
}

impl<F> From<F> for ClosureHook<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

impl<O, A, L, F> SimulationHook<O, A, L> for ClosureHook<F>
where
    L: Logger + ?Sized,
    F: FnMut(&Step<O, A>) -> bool,
{
    fn call(&mut self, step: &Step<O, A>, _: &mut L) -> bool {
        (self.f)(step)
    }
}

/// A hook that logs step and episode statistics.
#[derive(Debug)]
pub struct StepLogger<OS, AS> {
    pub observation_space: OS,
    pub action_space: AS,

    /// Length of the current episode
    episode_length: u64,
    episode_reward: f64,
}

impl<OS, AS> StepLogger<OS, AS> {
    pub fn new(observation_space: OS, action_space: AS) -> Self {
        Self {
            observation_space,
            action_space,
            episode_length: 0,
            episode_reward: 0.0,
        }
    }
}

impl<OS, AS, L> SimulationHook<OS::Element, AS::Element, L> for StepLogger<OS, AS>
where
    OS: ElementRefInto<Loggable>,
    AS: ElementRefInto<Loggable>,
    L: Logger + ?Sized,
{
    fn call(&mut self, step: &Step<OS::Element, AS::Element>, logger: &mut L) -> bool {
        use Event::*;
        logger.log(Step, "reward", step.reward.into()).unwrap();
        logger
            .log(
                Step,
                "observation",
                self.observation_space.elem_ref_into(&step.observation),
            )
            .unwrap();
        logger
            .log(
                Step,
                "action",
                self.action_space.elem_ref_into(&step.action),
            )
            .unwrap();
        logger.done(Event::Step);

        self.episode_length += 1;
        self.episode_reward += step.reward;
        if step.episode_done {
            logger
                .log(Episode, "length", (self.episode_length as f64).into())
                .unwrap();
            self.episode_length = 0;

            logger
                .log(Episode, "reward", self.episode_reward.into())
                .unwrap();
            self.episode_reward = 0.0;
            logger.done(Episode);
        }

        true
    }
}

// TODO: Generate for other tuple sizes with a macro
// Consider impl_trait_for_tuples crate

impl GenericSimulationHook for () {
    fn call<O, A, L: Logger + ?Sized>(&mut self, _: &Step<O, A>, _: &mut L) -> bool {
        true
    }
}

/// For a pair of hooks, continue if either allow continuing.
impl<O, A, L, T0, T1> SimulationHook<O, A, L> for (T0, T1)
where
    L: Logger + ?Sized,
    T0: SimulationHook<O, A, L>,
    T1: SimulationHook<O, A, L>,
{
    fn call(&mut self, step: &Step<O, A>, logger: &mut L) -> bool {
        let result_0 = self.0.call(step, logger);
        let result_1 = self.1.call(step, logger);
        result_0 && result_1
    }
}

/// Count occurrences of each action by index.
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
impl<O, AS: FiniteSpace, L: Logger + ?Sized> SimulationHook<O, AS::Element, L>
    for IndexedActionCounter<AS>
{
    fn call(&mut self, step: &Step<O, AS::Element>, _: &mut L) -> bool {
        self.counts[self.action_space.to_index(&step.action)] += 1;
        true
    }
}
