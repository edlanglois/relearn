mod actor_critic;
pub mod critics;
pub mod features;
pub mod policies;

pub use actor_critic::{ActorCriticAgent, ActorCriticConfig};

use crate::logging::StatsLogger;
use crate::torch::modules::{AsModule, Module};
use crate::torch::optimizers::{opt_expect_ok_log, Optimizer};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::time::Instant;
use tch::{Device, Tensor};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum ToLog {
    /// Don't log the absolute loss value (can log loss changes).
    NoAbsLoss,
    /// Log everything
    All,
}

/// Take n backward steps of a loss function with logging.
fn n_backward_steps<O, F, L>(
    optimizer: &mut O,
    mut loss_fn: F,
    n: u64,
    mut logger: L,
    to_log: ToLog,
    err_msg: &str,
) where
    O: Optimizer + ?Sized,
    F: FnMut() -> Tensor,
    L: StatsLogger,
{
    let mut step_logger = (&mut logger).with_scope("step");
    let mut prev_loss = None;
    let mut prev_start = Instant::now();
    for _ in 0..n {
        let result = optimizer.backward_step(&mut loss_fn, &mut step_logger);
        let loss = opt_expect_ok_log(result, err_msg).map(f64::from);

        if let Some(loss_improvement) = prev_loss.and_then(|p| loss.map(|l| p - l)) {
            step_logger.log_scalar("loss_improvement", loss_improvement);
        }
        prev_loss = loss;
        let end = Instant::now();
        step_logger.log_duration("time", end - prev_start);
        prev_start = end;
    }
    if matches!(to_log, ToLog::All) {
        if let Some(loss) = prev_loss {
            logger.log_scalar("loss", loss);
        }
    }
}

/// Wraps a module to have a lazily-initialized CPU copy if not already in CPU memory.
///
/// This is useful for models used both in training and in simulation because large batch size
/// training is most efficient on the GPU while batch-size-1 simulation is most efficient on the
/// CPU.
#[derive(Serialize, Deserialize)]
pub struct WithCpuCopy<T: AsModule> {
    inner: T,
    /// Device on which `policy` (the master copy) is stored.
    // Tensors will deserialize to CPU
    #[serde(skip, default = "cpu_device")]
    device: Device,
    #[serde(skip, default)]
    cpu_module: RefCell<Option<T::Module>>,
}

const fn cpu_device() -> Device {
    Device::Cpu
}

impl<T: AsModule> WithCpuCopy<T> {
    pub const fn new(inner: T, device: Device) -> Self {
        Self {
            inner,
            device,
            cpu_module: RefCell::new(None),
        }
    }
}

impl<T: AsModule + Clone> Clone for WithCpuCopy<T> {
    fn clone(&self) -> Self {
        Self::new(self.inner.clone(), self.device)
    }
}

impl<T: AsModule> AsModule for WithCpuCopy<T> {
    type Module = T::Module;

    #[inline]
    fn as_module(&self) -> &Self::Module {
        self.as_inner().as_module()
    }

    /// Get mutable reference to the module. Invalidates the cached CPU copy if any.
    #[inline]
    fn as_module_mut(&mut self) -> &mut Self::Module {
        self.as_inner_mut().as_module_mut()
    }
}

impl<T: AsModule> WithCpuCopy<T> {
    /// Get a reference to the inner struct.
    #[inline]
    pub const fn as_inner(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner struct. Invalidates the cached CPU copy if any.
    #[inline]
    pub fn as_inner_mut(&mut self) -> &mut T {
        self.cpu_module.take();
        &mut self.inner
    }

    /// Create a shallow clone of the module on CPU memory.
    ///
    /// If the module is not already on the CPU device then a cached deep copy is created on the
    /// CPU first. This cached copy is reused on future calls.
    #[inline]
    pub fn shallow_clone_module_cpu(&self) -> T::Module {
        if self.device == Device::Cpu {
            self.as_module().shallow_clone()
        } else {
            self.cpu_module
                .borrow_mut()
                .get_or_insert_with(|| self.as_module().clone_to_device(Device::Cpu))
                .shallow_clone()
        }
    }
}

impl<T: AsModule + fmt::Debug> fmt::Debug for WithCpuCopy<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("WithCpuCopy")
            .field("inner", &self.inner)
            .field("device", &self.device)
            .field(
                "cpu_module",
                &self.cpu_module.borrow().as_ref().map(|_| "..."),
            )
            .finish()
    }
}

impl<T: AsModule + PartialEq> PartialEq for WithCpuCopy<T> {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.inner == other.inner
    }
}
