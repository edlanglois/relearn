//! Definition structures
mod agent;
pub mod env;
mod optimizer;
mod policy;

pub use agent::{AgentDef, GaePolicyGradientAgentDef, PolicyGradientAgentDef};
pub use env::EnvDef;
pub use optimizer::OptimizerDef;
pub use policy::PolicyDef;
