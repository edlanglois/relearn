use super::agent::{
    EnvAnyAny, EnvFiniteFinite, EnvMetaFiniteFinite, For, RLActionSpace, RLObservationSpace,
};
use super::env::{VisitEnvAnyAny, VisitEnvBase, VisitEnvFiniteFinite, VisitEnvMetaFinitFinite};
use super::{AgentDef, EnvDef, HooksDef, MultithreadAgentDef};
use crate::envs::{BuildEnv, MetaObservationSpace};
use crate::logging::{BuildThreadLogger, TimeSeriesLogger};
use crate::simulation::{MultithreadSimulatorConfig, SerialSimulator, Simulator};
use crate::spaces::FiniteSpace;

/// Construct a boxed serial agent-environment simulator
///
/// [`Simulator::run_simulation`] will return an error if
/// the agent and environment are incompatible.
///
/// # Args
/// * `env_def`   - Environment definition
/// * `agent_def` - Agent definition
/// * `hooks_def` - Simulation hooks definition
pub fn boxed_serial_simulator(
    env_def: EnvDef,
    agent_def: AgentDef,
    hooks_def: HooksDef,
) -> Box<dyn Simulator> {
    env_def.visit(SerialSimulatorVisitor {
        agent_def,
        hooks_def,
    })
}

/// Construct a boxed multithread agent-environment simulator.
///
/// [`Simulator::run_simulation`] will return an error if
/// the agent and environment are incompatible.
///
/// # Args
/// * `sim_config` - Simulator configuration
/// * `env_def` - Environment definition
/// * `agent_def` - Multi-thread agent definition
/// * `hooks_def` - Simulation hooks definition
pub fn boxed_multithread_simulator<LC>(
    sim_config: MultithreadSimulatorConfig,
    env_def: EnvDef,
    agent_def: MultithreadAgentDef,
    hooks_def: HooksDef,
    worker_logger_config: LC,
) -> Box<dyn Simulator>
where
    LC: BuildThreadLogger + 'static,
    LC::ThreadLogger: TimeSeriesLogger + Send + 'static,
{
    env_def.visit(MultithreadSimulatorVisitor {
        sim_config,
        agent_def,
        hooks_def,
        worker_logger_config,
    })
}

/// Environment visitor that constructs a serial simulator.
struct SerialSimulatorVisitor {
    pub agent_def: AgentDef,
    pub hooks_def: HooksDef,
}

impl VisitEnvBase for SerialSimulatorVisitor {
    type Out = Box<dyn Simulator>;
}

impl VisitEnvFiniteFinite for SerialSimulatorVisitor {
    fn visit_env_finite_finite<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace + FiniteSpace,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace + FiniteSpace,
    {
        Box::new(SerialSimulator::new(
            env_config,
            For::<EnvFiniteFinite, _>::new(self.agent_def),
            self.hooks_def,
        ))
    }
}

impl VisitEnvMetaFinitFinite for SerialSimulatorVisitor {
    fn visit_env_meta_finite_finite<EC, OS, AS>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS> + 'static,
        OS: RLObservationSpace + FiniteSpace + Clone,
        OS::Element: Clone,
        AS: RLActionSpace + FiniteSpace + Clone,
        AS::Element: Clone,
        EC::ObservationSpace: RLObservationSpace,
        EC::Observation: Clone,
    {
        Box::new(SerialSimulator::new(
            env_config,
            For::<EnvMetaFiniteFinite, _>::new(self.agent_def),
            self.hooks_def,
        ))
    }
}

impl VisitEnvAnyAny for SerialSimulatorVisitor {
    fn visit_env_any_any<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace,
    {
        Box::new(SerialSimulator::new(
            env_config,
            For::<EnvAnyAny, _>::new(self.agent_def),
            self.hooks_def,
        ))
    }
}

/// Environment visitor that constructs a multithread simulator
struct MultithreadSimulatorVisitor<LC> {
    pub sim_config: MultithreadSimulatorConfig,
    pub agent_def: MultithreadAgentDef,
    pub hooks_def: HooksDef,
    pub worker_logger_config: LC,
}

impl<LC> VisitEnvBase for MultithreadSimulatorVisitor<LC> {
    type Out = Box<dyn Simulator>;
}

impl<LC> VisitEnvFiniteFinite for MultithreadSimulatorVisitor<LC>
where
    LC: BuildThreadLogger + 'static,
    LC::ThreadLogger: TimeSeriesLogger + Send + 'static,
{
    fn visit_env_finite_finite<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace + FiniteSpace,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace + FiniteSpace,
        EC::Environment: Send + 'static,
    {
        Box::new(self.sim_config.build_simulator(
            env_config,
            For::<EnvFiniteFinite, _>::new(self.agent_def),
            self.hooks_def,
            self.worker_logger_config,
        ))
    }
}

impl<LC> VisitEnvMetaFinitFinite for MultithreadSimulatorVisitor<LC>
where
    LC: BuildThreadLogger + 'static,
    LC::ThreadLogger: TimeSeriesLogger + Send + 'static,
{
    fn visit_env_meta_finite_finite<EC, OS, AS>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS> + 'static,
        EC::Environment: Send + 'static,
        OS: RLObservationSpace + FiniteSpace + Clone,
        OS::Element: Clone,
        AS: RLActionSpace + FiniteSpace + Clone,
        AS::Element: Clone,
        EC::ObservationSpace: RLObservationSpace,
        EC::Observation: Clone,
    {
        Box::new(self.sim_config.build_simulator(
            env_config,
            For::<EnvMetaFiniteFinite, _>::new(self.agent_def),
            self.hooks_def,
            self.worker_logger_config,
        ))
    }
}

impl<LC> VisitEnvAnyAny for MultithreadSimulatorVisitor<LC>
where
    LC: BuildThreadLogger + 'static,
    LC::ThreadLogger: TimeSeriesLogger + Send + 'static,
{
    fn visit_env_any_any<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace,
        EC::Environment: Send + 'static,
    {
        Box::new(self.sim_config.build_simulator(
            env_config,
            For::<EnvAnyAny, _>::new(self.agent_def),
            self.hooks_def,
            self.worker_logger_config,
        ))
    }
}
