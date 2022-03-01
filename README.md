# ReLearn: A Reinforcement Learning Library
A reinforcement learning library and experiment runner.
Uses pytorch as the neural network backend via the [tch](https://docs.rs/tch)
interface to the C++ API.

At the moment this is designed for personal use. It is in-development and
unstable so expect breaking changes with updates.

Read the documentation at <https://docs.rs/relearn>.

## Examples
### Chain Environment with Tabular Q Learning
```sh
cargo run --release --example chain-tabular-q
```
This environment has infinitely long episodes.

### Cart-Pole with Trust-Region Policy Optimization
```sh
cargo run --release --example cartpole-trpo
```
Uses a feed-forward MLP for the policy and a separate MLP for the critic
(baseline).
The displayed statistics are also saved to `data/cartpole-trpo/` and can be
viewed with `tensorboard --logdir data/cartpole-trpo`.
