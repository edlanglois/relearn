use super::agent::{
    ForAnyAny, ForFiniteFinite, ForMetaFiniteFinite, RLActionSpace, RLObservationSpace,
};
use super::env::{VisitEnvAnyAny, VisitEnvBase, VisitEnvFiniteFinite, VisitEnvMetaFinitFinite};
use super::{AgentDef, EnvDef, HooksDef, MultiThreadAgentDef};
use crate::envs::{BuildEnv, MetaObservationSpace};
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
pub fn boxed_multithread_simulator(
    sim_config: MultithreadSimulatorConfig,
    env_def: EnvDef,
    agent_def: MultiThreadAgentDef,
    hooks_def: HooksDef,
) -> Box<dyn Simulator> {
    env_def.visit(MultithreadSimulatorVisitor {
        sim_config,
        agent_def,
        hooks_def,
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
            ForFiniteFinite::new(self.agent_def),
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
            ForMetaFiniteFinite::new(self.agent_def),
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
            ForAnyAny::new(self.agent_def),
            self.hooks_def,
        ))
    }
}

/// Environment visitor that constructs a multithread simulator
struct MultithreadSimulatorVisitor {
    pub sim_config: MultithreadSimulatorConfig,
    pub agent_def: MultiThreadAgentDef,
    pub hooks_def: HooksDef,
}

impl VisitEnvBase for MultithreadSimulatorVisitor {
    type Out = Box<dyn Simulator>;
}

impl VisitEnvFiniteFinite for MultithreadSimulatorVisitor {
    fn visit_env_finite_finite<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace + FiniteSpace,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace + FiniteSpace,
        EC::Environment: Send + 'static,
    {
        self.sim_config.build_boxed_simulator(
            env_config,
            ForFiniteFinite::new(self.agent_def),
            self.hooks_def,
        )
    }
}

impl VisitEnvMetaFinitFinite for MultithreadSimulatorVisitor {
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
        self.sim_config.build_boxed_simulator(
            env_config,
            ForMetaFiniteFinite::new(self.agent_def),
            self.hooks_def,
        )
    }
}

impl VisitEnvAnyAny for MultithreadSimulatorVisitor {
    fn visit_env_any_any<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace,
        EC::Environment: Send + 'static,
    {
        self.sim_config.build_boxed_simulator(
            env_config,
            ForAnyAny::new(self.agent_def),
            self.hooks_def,
        )
    }
}
