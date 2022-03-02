use super::{BuildEnv, BuildEnvError, EnvStructure, Environment, Successor};
use crate::logging::StatsLogger;
use crate::spaces::{Indexed, IndexedTypeSpace, IntervalSpace};
use crate::Prng;
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

/// Configuration for the [`CartPole`] environment.
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct CartPoleConfig {
    /// Physics configuration
    pub physics_config: PhysicalConstants,
    /// Environment environment configuration
    pub env_config: EnvironmentParams,
}

impl BuildEnv for CartPoleConfig {
    type Observation = CartPolePhysicalState;
    type Action = Push;
    type ObservationSpace = CartPolePhysicalStateSpace;
    type ActionSpace = IndexedTypeSpace<Push>;
    type Environment = CartPole;

    fn build_env(&self, _: &mut Prng) -> Result<Self::Environment, BuildEnvError> {
        Ok(CartPole::new(self.physics_config, self.env_config))
    }
}

/// Cart-Pole environment
///
/// Consists of a simulated cart on a track with a vertical pole attached by a hinge on the top.
/// The goal is to keep the pole upright by applying left and right forces to the cart.
///
/// The environment is based on [Barto et al. (1983)][barto1983] with updated dynamics equations
/// from [Florian (2005)][florian2005], who corrects the friction term.
/// The default dynamics constants and episode parameters are based on the
/// [OpenAI Gym][gym_cartpole] [CartPole-v1 environment][cartpole_source].
///
/// [barto1983]: https://ieeexplore.ieee.org/document/6313077
/// [florian2005]: https://coneural.org/florian/papers/05_cart_pole.pdf
/// [gym_cartpole]: https://gym.openai.com/envs/CartPole-v1/
/// [cartpole_source]: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct CartPole {
    phys: InternalPhysicalConstants,
    env: EnvironmentParams,
}

impl CartPole {
    pub fn new(phys: PhysicalConstants, env: EnvironmentParams) -> Self {
        Self {
            phys: phys.into(),
            env,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Indexed, Serialize, Deserialize)]
pub enum Push {
    Left,
    Right,
}

impl EnvStructure for CartPole {
    type ObservationSpace = CartPolePhysicalStateSpace;
    type ActionSpace = IndexedTypeSpace<Push>;

    fn observation_space(&self) -> Self::ObservationSpace {
        let max_pos = self.env.max_pos;
        let max_angle = self.env.max_angle;
        CartPolePhysicalStateSpace {
            cart_position: IntervalSpace::new(-max_pos, max_pos),
            cart_velocity: IntervalSpace::default(),
            pole_angle: IntervalSpace::new(-max_angle, max_angle),
            pole_angular_velocity: IntervalSpace::default(),
        }
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexedTypeSpace::new()
    }

    fn reward_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn discount_factor(&self) -> f64 {
        self.env.discount_factor
    }
}

impl Environment for CartPole {
    type State = CartPoleInternalState;
    type Observation = CartPolePhysicalState;
    type Action = Push;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        // All parameters are sampled from the same range of values
        let dist = Uniform::new_inclusive(-0.05, 0.05);
        CartPoleInternalState {
            physical: CartPolePhysicalState {
                cart_position: dist.sample(rng),
                cart_velocity: dist.sample(rng),
                pole_angle: dist.sample(rng),
                pole_angular_velocity: dist.sample(rng),
            },
            cached_normal_velocity_is_positive: true,
        }
    }

    fn observe(&self, state: &Self::State, _: &mut Prng) -> Self::Observation {
        assert!(
            state.physical.cart_position >= -self.env.max_pos
                && state.physical.cart_position <= self.env.max_pos
                && state.physical.pole_angle >= -self.env.max_angle
                && state.physical.pole_angle <= self.env.max_angle,
            "out-of-bounds state should not have been produced"
        );
        state.physical
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        _: &mut Prng,
        _: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, f64) {
        let applied_force = match action {
            Push::Left => -self.env.action_force,
            Push::Right => self.env.action_force,
        };
        let next_state = self.phys.next_state(&state, applied_force);
        let reward = 1.0;
        let terminal = next_state.physical.cart_position.abs() > self.env.max_pos
            || next_state.physical.pole_angle.abs() > self.env.max_angle;
        // The OpenAI gym version returns the state as well as setting the done flag
        // but the gym API does not distinguish between the episode stopping with or without the
        // hypothetical potential for future rewards.
        // Here, the successor must be Terminate to indicate that all future rewards are 0.
        let successor = if terminal {
            Successor::Terminate
        } else {
            Successor::Continue(next_state)
        };
        (successor, reward)
    }
}

/// Physical constants for the [`CartPole`] environment.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalConstants {
    /// Downward force of gravity (m/s^2)
    pub gravity: f64,
    /// Mass of the cart (kg)
    pub mass_cart: f64,
    /// Mass of the pole (kg)
    pub mass_pole: f64,
    /// Half the length of the pole (m)
    pub length_half_pole: f64,
    /// Coefficient of friction between the cart and the track (unitless).
    ///
    /// The track is assumed to fully confine the cart in the vertical direction and this same
    /// friction coefficient applies whether the normal force of the cart is up or down.
    pub friction_cart: f64,
    /// Coefficient of friction between the pole and the cart at the hinge (unitless).
    pub friction_pole: f64,
    /// Simulation time step (s)
    pub time_step: f64,
}

impl Default for PhysicalConstants {
    fn default() -> Self {
        // Defaults (other than friction) from the OpenAI CartPole-v1 environment
        Self {
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            length_half_pole: 0.5,
            friction_cart: 0.01,
            friction_pole: 0.01,
            time_step: 0.02,
        }
    }
}

/// Parameters for [`CartPole`] as a reinforcement learning environment.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentParams {
    /// Magnitude of the force (N) applied by actions.
    pub action_force: f64,
    /// Maximum absolute position (meters) before the episode is ended.
    pub max_pos: f64,
    /// Maximum absolute pole angle from vertical (radians) before the episode is ended.
    pub max_angle: f64,
    /// Discount factor
    pub discount_factor: f64,
}

impl Default for EnvironmentParams {
    fn default() -> Self {
        // Defaults (except discount factor) from the OpenAI CartPole-v1 environment
        Self {
            action_force: 10.0,
            max_pos: 2.4,
            max_angle: 12.0f64.to_radians(), // 12 degrees
            discount_factor: 0.99,
        }
    }
}

/// Internal cart-pole constants with pre-computed common values.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
struct InternalPhysicalConstants {
    /// Fundamental constants
    c: PhysicalConstants,
    /// Gravitational weight of the combined system (N): `gravity * (mass_cart + mass_pole)`.
    total_weight: f64,
    /// `1 / (mass_cart + mass_pole)`,
    inv_total_mass: f64,
    /// `mass_pole * length_half_pole`
    mass_length_pole: f64,
}

impl Default for InternalPhysicalConstants {
    fn default() -> Self {
        PhysicalConstants::default().into()
    }
}

impl From<PhysicalConstants> for InternalPhysicalConstants {
    fn from(c: PhysicalConstants) -> Self {
        let total_mass = c.mass_cart + c.mass_pole;
        let total_weight = c.gravity * total_mass;
        let inv_total_mass = total_mass.recip();
        let mass_length_pole = c.mass_pole * c.length_half_pole;
        Self {
            c,
            total_weight,
            inv_total_mass,
            mass_length_pole,
        }
    }
}

/// Physical state of the [`CartPole`] environment.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct CartPolePhysicalState {
    /// Cart position from the track midpoint (m).
    pub cart_position: f64,
    /// Cart velocity (m/s).
    pub cart_velocity: f64,
    /// Angle of the pole from vertical (radians).
    pub pole_angle: f64,
    /// Pole angular velocity about the hinge (radians / s).
    pub pole_angular_velocity: f64,
}

/// [`CartPole`] physical state space.
#[derive(Debug, Copy, Clone, PartialEq, ProductSpace, Serialize, Deserialize)]
#[element(CartPolePhysicalState)]
pub struct CartPolePhysicalStateSpace {
    /// Cart position from the track midpoint (m).
    pub cart_position: IntervalSpace,
    /// Cart velocity (m/s).
    pub cart_velocity: IntervalSpace,
    /// Angle of the pole from vertical (radians).
    pub pole_angle: IntervalSpace,
    /// Pole angular velocity about the hinge (radians / s).
    pub pole_angular_velocity: IntervalSpace,
}

/// State of the [`CartPole`] environment.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct CartPoleInternalState {
    /// Physical state.
    physical: CartPolePhysicalState,

    /// Cached sign of normal_force * cart_velocity.
    ///
    /// This is an intermediate term in the dynamics equations.
    /// The equations are slightly circular and the calculation for this term depends on its own
    /// value. Fortunately there are only two possible values and the value is likely to stay the
    /// same from one time step to the next.
    ///
    /// Therefore, the value from the previous time step is used and if the result is
    /// self-inconsistent then the negated value is used.
    cached_normal_velocity_is_positive: bool,
}

impl InternalPhysicalConstants {
    /// Simulate the state for one time step with an applied force on the cart (in N).
    pub fn next_state(
        &self,
        state: &CartPoleInternalState,
        applied_force: f64,
    ) -> CartPoleInternalState {
        // Reference:
        // "Correct equations for the dynamics of the cart-pole system" by Florian (2005)

        // Physical state
        let phys = &state.physical;

        let mut signed_cart_friction = if state.cached_normal_velocity_is_positive {
            self.c.friction_cart
        } else {
            -self.c.friction_cart
        };
        let (sin_angle, cos_angle) = phys.pole_angle.sin_cos();
        let angular_velocity_squared = phys.pole_angular_velocity * phys.pole_angular_velocity;

        let mut angular_acceleration = self.angular_acceleration(
            phys,
            applied_force,
            signed_cart_friction,
            angular_velocity_squared,
            sin_angle,
            cos_angle,
        );
        let mut normal_force = self.normal_force(
            angular_acceleration,
            angular_velocity_squared,
            sin_angle,
            cos_angle,
        );
        let normal_velocity_is_positive = (normal_force * phys.cart_velocity).is_sign_positive();

        if normal_velocity_is_positive != state.cached_normal_velocity_is_positive {
            // Re-calculate angular acceleration and normal force with the new signed friction
            signed_cart_friction = -signed_cart_friction;
            angular_acceleration = self.angular_acceleration(
                phys,
                applied_force,
                signed_cart_friction,
                angular_velocity_squared,
                sin_angle,
                cos_angle,
            );
            normal_force = self.normal_force(
                angular_acceleration,
                angular_velocity_squared,
                sin_angle,
                cos_angle,
            );
            // Not sure if it is possible for the normal force to change sign again in which case
            // the there would be no consistent solution. Florian does not say to check.
        }

        // Calculate horizontal acceleration of the cart (m/s^2)
        // Force of the pole on the cart
        let force_pole = self.mass_length_pole
            * (angular_velocity_squared * sin_angle + angular_acceleration * cos_angle);
        // Force of friction on the cart
        let force_friction = -signed_cart_friction * normal_force;
        let net_force = applied_force + force_pole + force_friction;
        let cart_acceleration = net_force * self.inv_total_mass;

        // Update state with semi-implicit euler integration
        let cart_velocity = phys.cart_velocity + self.c.time_step * cart_acceleration;
        let cart_position = phys.cart_position + self.c.time_step * cart_velocity;
        let pole_angular_velocity =
            phys.pole_angular_velocity + self.c.time_step * angular_acceleration;
        let pole_angle = phys.pole_angle + self.c.time_step * phys.pole_angular_velocity;

        CartPoleInternalState {
            physical: CartPolePhysicalState {
                cart_velocity,
                cart_position,
                pole_angular_velocity,
                pole_angle,
            },
            cached_normal_velocity_is_positive: normal_velocity_is_positive,
        }
    }

    /// The pole angular acceleration
    ///
    /// # Args
    /// * `applied_force`            - Applied horizontal force on the cart (N).
    /// * `signed_cart_friction`     - Signed cart friction coefficient:
    ///                                    `friction_cart * sign(normal_force * cart_velocity)`
    /// * `angular_velocity_squared` - `pole_angular_velocity ** 2`
    /// * `sin_angle`                - `sin(pole_angle)`.
    /// * `cos_angle`                - `cos(pole_angle)`.
    fn angular_acceleration(
        &self,
        state: &CartPolePhysicalState,
        applied_force: f64,
        signed_cart_friction: f64,
        angular_velocity_squared: f64,
        sin_angle: f64,
        cos_angle: f64,
    ) -> f64 {
        // Reference:
        // "Correct equations for the dynamics of the cart-pole system" by Florian (2005)
        //
        // Decompose equation (21) as
        // numerator / denominator where
        // numerator = (g*sin(theta) + cos(theta)*(alpha + g*signed_cart_friction) - beta)

        let alpha = (-applied_force
            - self.mass_length_pole
                * angular_velocity_squared
                * (sin_angle + signed_cart_friction * cos_angle))
            * self.inv_total_mass;
        let beta = self.c.friction_pole * state.pole_angular_velocity / self.mass_length_pole;
        let numerator = self.c.gravity * sin_angle
            + cos_angle * (alpha + self.c.gravity * signed_cart_friction)
            - beta;

        let denominator = self.c.length_half_pole
            * (4.0 / 3.0
                - self.c.mass_pole
                    * cos_angle
                    * self.inv_total_mass
                    * (cos_angle - signed_cart_friction));
        numerator / denominator
    }

    /// Normal force of the cart against the track (N).
    ///
    /// Positive for downward normal force and negative for upward.
    fn normal_force(
        &self,
        angular_acceleration: f64,
        angular_velocity_squared: f64,
        sin_angle: f64,
        cos_angle: f64,
    ) -> f64 {
        self.total_weight
            - self.mass_length_pole
                * (angular_acceleration * sin_angle + angular_velocity_squared * cos_angle)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing;
    use super::*;

    #[test]
    fn run_default() {
        testing::check_structured_env(&CartPole::default(), 1000, 0);
    }
}
