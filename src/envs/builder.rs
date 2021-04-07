use rand::distributions::BernoulliError;
use thiserror::Error;

pub trait EnvBuilder<E> {
    /// Build an environment instance.
    ///
    /// # Args
    /// * `seed` - Seed for pseudo-randomness used by the environment.
    ///     Includes both randomization of the environment structure, and
    ///     random sampling of step outcomes within this structure.
    fn build(&self, seed: u64) -> Result<E, BuildEnvError>;
}

/// Error building an environment
#[derive(Debug, Error)]
pub enum BuildEnvError {
    #[error(transparent)]
    BernoulliError(#[from] BernoulliError),
}
