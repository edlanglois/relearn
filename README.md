# Reinforcement Learning with Rust

## Saved Runs
### RL Squared Training
Set `steps-per-epoch` to whatever value fits on the GPU so long as data
collection does not take more than about 15s.
```sh
cargo run --release -- meta-uniform-bernoulli-bandits trpo \
    --num-actions 5 \
    --episodes-per-trial 10 \
    --gae-discount-factor 0.99 \
    --gae-lambda 0.3 \
    --steps-per-epoch 25000 \
    --policy gru-mlp \
    --step-value gae \
    --rnn-hidden-size 256 \
    --hidden-sizes \
    --device cuda
```
