use super::super::seq_modules::{IterativeModule, WithState};
use super::super::ModuleBuilder;
use tch::nn::Path;

impl<T, MB> ModuleBuilder<WithState<T>> for MB
where
    MB: ModuleBuilder<T>,
    T: IterativeModule,
{
    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> WithState<T> {
        self.build_module(vs, in_dim, out_dim).into()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::seq_modules::SeqModRnn;
    use super::super::{MlpConfig, RnnConfig};
    use super::*;
    use tch::{nn, Device};

    #[test]
    fn linear_builds() {
        let config = MlpConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: WithState<nn::Sequential> = config.build_module(&vs.root(), 1, 1);
    }

    #[test]
    fn gru_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: WithState<SeqModRnn<nn::GRU>> = config.build_module(&vs.root(), 1, 1);
    }
}
