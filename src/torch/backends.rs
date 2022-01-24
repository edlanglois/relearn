//! Details concerning torch backends.
use tch::Cuda;

/// A RAII guard to enable / disable cuDNN and restore the previous state after being dropped.
pub struct WithCudnnEnabled {
    previous_state: bool,
}

impl WithCudnnEnabled {
    pub fn new(enabled: bool) -> Self {
        let previous_state = Cuda::user_enabled_cudnn();
        Cuda::set_user_enabled_cudnn(enabled);
        Self { previous_state }
    }
}

impl Drop for WithCudnnEnabled {
    fn drop(&mut self) {
        Cuda::set_user_enabled_cudnn(self.previous_state);
    }
}
