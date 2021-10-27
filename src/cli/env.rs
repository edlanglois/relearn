//! Parse environment definition from Options
use super::{Options, Update, WithUpdate};
use crate::defs::{env::DistributionType, BanditMeanRewards, EnvDef};
use crate::envs::{
    Chain, DirichletRandomMdps, FirstPlayerView, FruitGame, MemoryGame, MetaPomdp, OneHotBandits,
    StepLimit, UniformBernoulliBandits, Wrapped,
};
use clap::ArgEnum;

/// Environment name
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum EnvType {
    DeterministicBandit,
    BernoulliBandit,
    Chain,
    Fruit,
    MemoryGame,
    MetaOneHotBandits,
    MetaUniformBernoulliBandits,
    MetaDirichletMdps,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum BanditArmPrior {
    Fixed,
    Uniform,
}

impl From<&Options> for EnvDef {
    fn from(opts: &Options) -> Self {
        use EnvType::*;
        match opts.environment {
            DeterministicBandit => bandit_env_def(DistributionType::Deterministic, opts),
            BernoulliBandit => bandit_env_def(DistributionType::Bernoulli, opts),
            Chain => Self::Chain(opts.into()),
            Fruit => Self::Fruit(opts.into()),
            MemoryGame => Self::MemoryGame(opts.into()),
            MetaOneHotBandits => Self::MetaOneHotBandits(opts.into()),
            MetaUniformBernoulliBandits => Self::MetaUniformBernoulliBandits(opts.into()),
            MetaDirichletMdps => Self::MetaDirichletMdps(opts.into()),
        }
    }
}

fn bandit_env_def(distribution: DistributionType, opts: &Options) -> EnvDef {
    match opts.arm_prior {
        BanditArmPrior::Fixed => EnvDef::Bandit(distribution, opts.into()),
        BanditArmPrior::Uniform => EnvDef::Bandit(distribution, opts.into()),
    }
}

impl From<&Options> for BanditMeanRewards {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for BanditMeanRewards {
    fn update(&mut self, opts: &Options) {
        if let Some(mean_rewards) = &opts.arm_rewards {
            self.mean_rewards = mean_rewards.clone()
        }
    }
}

impl From<&Options> for Chain {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for Chain {
    fn update(&mut self, opts: &Options) {
        if let Some(num_states) = opts.num_states {
            self.size = num_states.try_into().unwrap();
        }
        if let Some(discount_factor) = opts.discount_factor {
            self.discount_factor = discount_factor;
        }
    }
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> From<&Options>
    for FruitGame<W, H, VW, VH>
{
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> Update<&Options>
    for FruitGame<W, H, VW, VH>
{
    fn update(&mut self, _opts: &Options) {}
}

impl<'a, E> From<&'a Options> for FirstPlayerView<E>
where
    E: From<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::new(opts.into())
    }
}

impl<'a, E> Update<&'a Options> for FirstPlayerView<E>
where
    E: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.inner.update(opts)
    }
}

impl From<&Options> for MemoryGame {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for MemoryGame {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_actions = num_actions.try_into().unwrap();
        }
        if let Some(episode_len) = opts.episode_len {
            self.history_len = (episode_len - 1).try_into().unwrap();
        }
    }
}

impl<'a, E> From<&'a Options> for MetaPomdp<E>
where
    Self: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, E> Update<&'a Options> for MetaPomdp<E>
where
    E: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.env_distribution.update(opts);
        if let Some(episodes_per_trial) = opts.episodes_per_trial {
            self.episodes_per_trial = episodes_per_trial;
        }
    }
}

impl<'a, T, W> From<&'a Options> for Wrapped<T, W>
where
    T: From<&'a Options>,
    W: From<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self {
            inner: opts.into(),
            wrapper: opts.into(),
        }
    }
}

impl<'a, T, W> Update<&'a Options> for Wrapped<T, W>
where
    T: Update<&'a Options>,
    W: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.inner.update(opts);
        self.wrapper.update(opts);
    }
}

impl From<&Options> for StepLimit {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for StepLimit {
    fn update(&mut self, opts: &Options) {
        if let Some(max_steps_per_episode) = opts.max_steps_per_episode {
            self.max_steps_per_episode = max_steps_per_episode;
        }
    }
}

impl From<&Options> for UniformBernoulliBandits {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for UniformBernoulliBandits {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_arms = num_actions.try_into().unwrap();
        }
    }
}

impl From<&Options> for OneHotBandits {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for OneHotBandits {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_arms = num_actions.try_into().unwrap();
        }
    }
}

impl From<&Options> for DirichletRandomMdps {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for DirichletRandomMdps {
    fn update(&mut self, opts: &Options) {
        if let Some(num_states) = opts.num_states {
            self.num_states = num_states.try_into().unwrap();
        }
        if let Some(num_actions) = opts.num_actions {
            self.num_actions = num_actions.try_into().unwrap();
        }
        if let Some(discount_factor) = opts.discount_factor {
            self.discount_factor = discount_factor;
        }
    }
}
