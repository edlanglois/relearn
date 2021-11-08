use super::agent::{RLActionSpace, RLObservationSpace};
use crate::envs::{
    Bandit, BuildEnv, BuildEnvError, BuildPomdp, Chain as ChainEnv, DirichletRandomMdps,
    EnvStructure, FirstPlayerView, FruitGame, MemoryGame as MemoryGameEnv, MetaObservationSpace,
    MetaPomdp, OneHotBandits, Pomdp, UniformBernoulliBandits, WithStepLimit,
};
use crate::spaces::{FiniteSpace, Space};
use crate::utils::distributions::{Bernoulli, Bounded, Deterministic, FromMean};
use rand::distributions::Distribution;
use std::borrow::Borrow;
use std::error::Error;
use std::marker::PhantomData;

/// Environment definition
#[derive(Debug, Clone)]
pub enum EnvDef {
    /// Multi-armed bandit with fixed arm distributions.
    Bandit(DistributionType, BanditMeanRewards),
    /// The Chain environment,
    Chain(ChainEnv),
    /// Fruit game (first player with second player doing nothing),
    Fruit(WithStepLimit<FirstPlayerView<FruitGame<5, 5, 5, 5>>>),
    /// The Memory Game environment
    MemoryGame(MemoryGameEnv),
    /// Meta one-hot bandits environment
    MetaOneHotBandits(MetaPomdp<OneHotBandits>),
    /// Meta uniform bernoulli bandits environment
    MetaUniformBernoulliBandits(MetaPomdp<UniformBernoulliBandits>),
    /// Meta dirichlet MDPs environment
    MetaDirichletMdps(MetaPomdp<WithStepLimit<DirichletRandomMdps>>),
}

/// Definition of a scalar floating-point distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionType {
    Deterministic,
    Bernoulli,
}

/// Bandit mean arm reads
#[derive(Debug, Clone, PartialEq)]
pub struct BanditMeanRewards {
    pub mean_rewards: Vec<f64>,
}

impl Default for BanditMeanRewards {
    fn default() -> Self {
        Self {
            mean_rewards: vec![0.2, 0.8],
        }
    }
}

#[derive(Debug, Clone)]
struct BanditConfig<D> {
    mean_rewards: Vec<f64>,
    // `fn() -> D` is used so that BanditConfig is Send & Sync
    // Reference: https://stackoverflow.com/a/50201389/1267562
    distribution: PhantomData<fn() -> D>,
}

impl<D> From<BanditMeanRewards> for BanditConfig<D> {
    fn from(config: BanditMeanRewards) -> Self {
        Self {
            mean_rewards: config.mean_rewards,
            distribution: PhantomData,
        }
    }
}

impl<D: FromMean<f64>> BuildPomdp for BanditConfig<D>
where
    D: FromMean<f64> + Distribution<f64> + Bounded<f64>,
    D::Error: Error + 'static,
{
    type State = <Self::Pomdp as Pomdp>::State;
    type Observation = <Self::Pomdp as Pomdp>::Observation;
    type Action = <Self::Pomdp as Pomdp>::Action;
    type ObservationSpace = <Self::Pomdp as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::Pomdp as EnvStructure>::ActionSpace;
    type Pomdp = Bandit<D>;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        let mean_rewards: &[f64] = self.mean_rewards.borrow();
        Bandit::from_means(mean_rewards).map_err(|e| BuildEnvError::Boxed(Box::new(e)))
    }
}

impl EnvDef {
    /// Transform according to a visitor.
    pub fn visit<T>(self, visitor: T) -> T::Out
    where
        T: VisitEnvFiniteFinite + VisitEnvMetaFinitFinite + VisitEnvAnyAny,
    {
        use EnvDef::*;
        match self {
            Bandit(dist_type, means) => match dist_type {
                DistributionType::Deterministic => {
                    let config = BanditConfig::<Deterministic<f64>>::from(means);
                    visitor.visit_env_finite_finite(config)
                }
                DistributionType::Bernoulli => {
                    let config = BanditConfig::<Bernoulli>::from(means);
                    visitor.visit_env_finite_finite(config)
                }
            },
            Chain(config) => visitor.visit_env_finite_finite(config),
            Fruit(config) => visitor.visit_env_finite_finite(config),
            MemoryGame(config) => visitor.visit_env_finite_finite(config),
            MetaOneHotBandits(config) => visitor.visit_env_meta_finite_finite(config),
            MetaUniformBernoulliBandits(config) => visitor.visit_env_meta_finite_finite(config),
            MetaDirichletMdps(config) => visitor.visit_env_meta_finite_finite(config),
        }
    }
}

/// Base trait that defines the output type of `VisitEnv*` visitors.
pub trait VisitEnvBase {
    type Out;
}

/// Visit an environment configuration with finite action and observation spaces.
pub trait VisitEnvFiniteFinite: VisitEnvBase {
    /// Transform into another type given an environment configuration.
    ///
    /// The environment configuration must build an environment with
    /// finite observation and action spaces.
    fn visit_env_finite_finite<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace + FiniteSpace,
        // TODO: Compiler can't tell these are the same
        <EC::ObservationSpace as Space>::Element: Clone,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace + FiniteSpace,
        <EC::ActionSpace as Space>::Element: Clone,
        EC::Environment: Send + 'static;
}

/// Visit a meta environment configuration with finite inner action and observation spaces.
pub trait VisitEnvMetaFinitFinite: VisitEnvBase {
    fn visit_env_meta_finite_finite<EC, OS, AS>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS> + 'static,
        EC::Environment: Send + 'static,
        OS: RLObservationSpace + FiniteSpace,
        // TODO: Compiler can't tell these are the same
        OS::Element: Clone,
        AS: RLActionSpace + FiniteSpace,
        AS::Element: Clone,
        // See impl BuildAgent for ForMetaFiniteFinite
        EC::ObservationSpace: RLObservationSpace,
        EC::Observation: Clone;
}

/// Visit an environment configuration for any reinforcement learning environment.
pub trait VisitEnvAnyAny: VisitEnvBase {
    fn visit_env_any_any<EC>(self, env_config: EC) -> Self::Out
    where
        EC: BuildEnv + 'static,
        EC::ObservationSpace: RLObservationSpace,
        // TODO: Compiler can't tell these are the same
        <EC::ObservationSpace as Space>::Element: Clone,
        EC::Observation: Clone,
        EC::ActionSpace: RLActionSpace,
        <EC::ActionSpace as Space>::Element: Clone,
        EC::Environment: Send + 'static;
}
