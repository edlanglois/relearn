use crate::envs::EnvStructure;
use crate::logging::{Loggable, TimeSeriesLogger};
use crate::simulation::hooks::{
    BuildSimulationHook, EpisodeLimitConfig, SimulationHook, StepLimitConfig, StepLoggerConfig,
};
use crate::simulation::FullStep;
use crate::spaces::ElementRefInto;
use std::ops::DerefMut;

/// Simulation hook definition.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum HookDef {
    StepLimit(StepLimitConfig),
    EpisodeLimit(EpisodeLimitConfig),
    StepLogger(StepLoggerConfig),
}

impl From<StepLimitConfig> for HookDef {
    fn from(config: StepLimitConfig) -> Self {
        Self::StepLimit(config)
    }
}

impl From<EpisodeLimitConfig> for HookDef {
    fn from(config: EpisodeLimitConfig) -> Self {
        Self::EpisodeLimit(config)
    }
}

impl From<StepLoggerConfig> for HookDef {
    fn from(config: StepLoggerConfig) -> Self {
        Self::StepLogger(config)
    }
}

/// Simulation hooks definition.
#[derive(Debug, Clone, PartialEq)]
pub struct HooksDef {
    pub hooks: Vec<HookDef>,
}

impl HooksDef {
    pub const fn new(hooks: Vec<HookDef>) -> Self {
        Self { hooks }
    }
}

impl From<Vec<HookDef>> for HooksDef {
    fn from(hooks: Vec<HookDef>) -> Self {
        Self { hooks }
    }
}

impl FromIterator<HookDef> for HooksDef {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = HookDef>,
    {
        Self {
            hooks: iter.into_iter().collect(),
        }
    }
}

impl<OS, AS> BuildSimulationHook<OS, AS> for HookDef
where
    OS: ElementRefInto<Loggable> + Send + 'static,
    AS: ElementRefInto<Loggable> + Send + 'static,
{
    type Hook = Box<dyn SimulationHook<OS::Element, AS::Element> + Send>;

    fn build_hook(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        num_threads: usize,
        thread_index: usize,
    ) -> Self::Hook {
        use HookDef::*;
        match self {
            StepLimit(config) => Box::new(config.build_hook(env, num_threads, thread_index)),
            EpisodeLimit(config) => Box::new(config.build_hook(env, num_threads, thread_index)),
            StepLogger(config) => Box::new(config.build_hook(env, num_threads, thread_index)),
        }
    }
}

impl<O, A> SimulationHook<O, A> for Box<dyn SimulationHook<O, A> + Send> {
    fn start(&mut self, logger: &mut dyn TimeSeriesLogger) -> bool {
        self.as_mut().start(logger)
    }

    fn call(&mut self, step: &FullStep<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool {
        self.as_mut().call(step, logger)
    }
}

impl<OS, AS> BuildSimulationHook<OS, AS> for HooksDef
where
    OS: ElementRefInto<Loggable> + Send + 'static,
    AS: ElementRefInto<Loggable> + Send + 'static,
{
    type Hook = Vec<Box<dyn SimulationHook<OS::Element, AS::Element> + Send>>;

    fn build_hook(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        num_threads: usize,
        thread_index: usize,
    ) -> Self::Hook {
        self.hooks
            .iter()
            .map(|h| h.build_hook(env, num_threads, thread_index))
            .collect()
    }
}

impl<O, A> SimulationHook<O, A> for Vec<Box<dyn SimulationHook<O, A> + Send>> {
    fn start(&mut self, logger: &mut dyn TimeSeriesLogger) -> bool {
        self.deref_mut().start(logger)
    }

    fn call(&mut self, step: &FullStep<O, A>, logger: &mut dyn TimeSeriesLogger) -> bool {
        self.deref_mut().call(step, logger)
    }
}
