use indicatif::ProgressBar;
use rust_rl::agents::RandomAgent;
use rust_rl::envs::{BernoulliBandit, EnvSpec};
use rust_rl::simulator;

fn main() {
    let seed = 1;
    let mut environment = BernoulliBandit::new(vec![0.2, 0.8], seed);
    println!("Environment: {}", environment);
    let mut agent = RandomAgent::new(environment.action_space(), seed + 1);
    println!("Agent: {}", agent);

    let num_steps = 1_000_000;
    let progress_bar = ProgressBar::new(num_steps);
    progress_bar.set_draw_delta(num_steps / 100); // Redraw every 1% of progress

    let mut step_count = 0;
    simulator::run(&mut environment, &mut agent, &mut |_step| {
        progress_bar.inc(1);
        step_count += 1;
        step_count >= num_steps
    });
    progress_bar.finish();
}
