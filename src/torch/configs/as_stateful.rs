use super::super::seq_modules::{AsStatefulIterator, IterativeModule};
use super::super::ModuleBuilder;
use std::borrow::Borrow;
use std::marker::PhantomData;
use tch::nn::Path;

pub struct AsStatefulIterConfig<M, T = M>(T, PhantomData<M>);

impl<M> Default for AsStatefulIterConfig<M>
where
    M: Default,
{
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

impl<M> From<M> for AsStatefulIterConfig<M, M> {
    fn from(module: M) -> Self {
        Self(module, PhantomData)
    }
}

impl<'a, M> From<&'a M> for AsStatefulIterConfig<M, &'a M> {
    fn from(module: &'a M) -> Self {
        Self(module, PhantomData)
    }
}

impl<M, T> ModuleBuilder for AsStatefulIterConfig<M, T>
where
    M: ModuleBuilder,
    <M as ModuleBuilder>::Module: IterativeModule,
    T: Borrow<M>,
{
    type Module = AsStatefulIterator<M::Module>;

    fn build<'a, U: Borrow<Path<'a>>>(
        &self,
        vs: U,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module {
        self.0.borrow().build(vs, input_dim, output_dim).into()
    }
}
