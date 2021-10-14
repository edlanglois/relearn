use super::Options;
use crate::defs::HooksDef;
use crate::simulation::hooks::{EpisodeLimitConfig, StepLimitConfig, StepLoggerConfig};

impl From<&Options> for HooksDef {
    fn from(opts: &Options) -> Self {
        let mut hooks = vec![StepLoggerConfig.into()];
        if let Some(max_steps) = opts.max_steps {
            hooks.push(StepLimitConfig::new(max_steps).into())
        }
        if let Some(max_episodes) = opts.max_episodes {
            hooks.push(EpisodeLimitConfig::new(max_episodes).into())
        }
        Self::new(hooks)
    }
}
