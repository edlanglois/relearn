# Reinforcement Learning with Rust

## Saved Runs
### RL Squared Training
Replications of some results from the paper
RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning
by Duan et al. https://arxiv.org/pdf/1611.02779.pdf

#### Bandits - TRPO RL2
Set `steps-per-epoch` to whatever value fits on the GPU so long as data
collection does not take more than about 15s.
```sh
cargo run --release -- \
    meta-uniform-bernoulli-bandits \
    trpo \
    --num-actions 5 \
    --episodes-per-trial 10 \
    --gae-discount-factor 0.99 \
    --gae-lambda 0.3 \
    --steps-per-epoch 25000 \
    --policy gru-mlp \
    --critic gae \
    --rnn-hidden-size 256 \
    --hidden-sizes \
    --device cuda
```
By 50 epochs it should be near the optimal value of 6.6 or 6.7
(the paper shows 6.7 for UCB1 but I get 6.6).

#### Bandits - UCB Baseline
```
cargo run --release -- \
    meta-uniform-bernoulli-bandits \
    resetting-meta:ucb1 \
    --num-actions 5 \
    --episodes-per-trial 10 \
```
