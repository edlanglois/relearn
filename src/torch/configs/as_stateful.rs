use super::super::seq_modules::{AsStatefulIterator, IterativeModule};
use super::super::ModuleBuilder;
use std::borrow::Borrow;
use tch::nn::Path;

impl<M, MB> ModuleBuilder<AsStatefulIterator<M>> for MB
where
    MB: ModuleBuilder<M>,
    M: IterativeModule,
{
    fn build<'a, U: Borrow<Path<'a>>>(
        &self,
        vs: U,
        input_dim: usize,
        output_dim: usize,
    ) -> AsStatefulIterator<M> {
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
        let _: AsStatefulIterator<nn::Sequential> = config.build(vs.root(), 1, 1);
    }

    #[test]
    fn gru_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: AsStatefulIterator<SeqModRnn<nn::GRU>> = config.build(vs.root(), 1, 1);
    }
}
