use super::super::seq_modules::{IterativeModule, WithState};
use super::super::ModuleBuilder;
use tch::nn::Path;

impl<T, MB> ModuleBuilder<WithState<T>> for MB
where
    MB: ModuleBuilder<T>,
    T: IterativeModule,
{
    fn build(&self, vs: &Path, input_dim: usize, output_dim: usize) -> WithState<T> {
        self.build(vs, input_dim, output_dim).into()
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
        let _: WithState<nn::Sequential> = config.build(&vs.root(), 1, 1);
    }

    #[test]
    fn gru_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: WithState<SeqModRnn<nn::GRU>> = config.build(&vs.root(), 1, 1);
    }
}
