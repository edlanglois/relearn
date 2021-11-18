use crate::envs::{BuildEnvError, BuildPomdp, EnvStructure, Environment, Pomdp};
use crate::logging::Logger;
use crate::spaces::{ProductSpace, Space};
use rand::rngs::StdRng;

/// Wraps a two-player game as a one-player game for the first player.
///
/// The second player always takes the default action.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FirstPlayerView<E> {
    pub inner: E,
}

impl<E> FirstPlayerView<E> {
    pub const fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E, OS1, OS2, AS1, AS2> BuildPomdp for FirstPlayerView<E>
where
    E: BuildPomdp<
        // Observation and Action values are implied {Observation,Action}Space
        // values but are not inferred automatically.
        Observation = (OS1::Element, OS2::Element),
        Action = (AS1::Element, AS2::Element),
        ObservationSpace = ProductSpace<(OS1, OS2)>,
        ActionSpace = ProductSpace<(AS1, AS2)>,
    >,
    OS1: Space,
    OS2: Space,
    AS1: Space,
    AS1::Element: Clone,
    AS2: Space,
    AS2::Element: Default,
{
    type State = E::State;
    type Observation = OS1::Element;
    type Action = AS1::Element;
    type ObservationSpace = OS1;
    type ActionSpace = AS1;
    type Pomdp = FirstPlayerView<E::Pomdp>;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        Ok(FirstPlayerView {
            inner: self.inner.build_pomdp()?,
        })
    }
}

impl<E, OS1, OS2, AS1, AS2> EnvStructure for FirstPlayerView<E>
where
    E: EnvStructure<
        ObservationSpace = ProductSpace<(OS1, OS2)>,
        ActionSpace = ProductSpace<(AS1, AS2)>,
    >,
    OS1: Space,
    AS1: Space,
{
    type ObservationSpace = OS1;
    type ActionSpace = AS1;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space().inner_spaces.0
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space().inner_spaces.0
    }
    fn reward_range(&self) -> (f64, f64) {
        self.inner.reward_range()
    }
    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<E, O1, O2, A1, A2> Pomdp for FirstPlayerView<E>
where
    E: Pomdp<Observation = (O1, O2), Action = (A1, A2)>,
    A1: Clone,
    A2: Default,
{
    type State = E::State;
    type Observation = O1;
    type Action = A1;

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        self.inner.initial_state(rng)
    }

    fn observe(&self, state: &Self::State, rng: &mut StdRng) -> Self::Observation {
        self.inner.observe(state, rng).0
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
        logger: &mut dyn Logger,
    ) -> (Option<Self::State>, f64, bool) {
        let joint_action = (action.clone(), Default::default());
        self.inner.step(state, &joint_action, rng, logger)
    }
}

impl<E, O1, O2, A1, A2> Environment for FirstPlayerView<E>
where
    E: Environment<Observation = (O1, O2), Action = (A1, A2)>,
    A1: Clone,
    A2: Default,
{
    type Observation = O1;
    type Action = A1;

    fn step(
        &mut self,
        action: &Self::Action,
        logger: &mut dyn Logger,
    ) -> (Option<Self::Observation>, f64, bool) {
        let joint_action = (action.clone(), Default::default());
        let (joint_observation, reward, episode_done) = self.inner.step(&joint_action, logger);
        let observation = joint_observation.map(|o| o.0);
        (observation, reward, episode_done)
    }

    fn reset(&mut self) -> Self::Observation {
        self.inner.reset().0
    }
}

/// Wraps a two-player game as a one-player game for the second player.
///
/// The first player always takes the default action.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SecondPlayerView<E> {
    pub inner: E,
}

impl<E> SecondPlayerView<E> {
    pub const fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E, OS1, OS2, AS1, AS2> BuildPomdp for SecondPlayerView<E>
where
    E: BuildPomdp<
        // Observation and Action values are implied {Observation,Action}Space
        // values but are not inferred automatically.
        Observation = (OS1::Element, OS2::Element),
        Action = (AS1::Element, AS2::Element),
        ObservationSpace = ProductSpace<(OS1, OS2)>,
        ActionSpace = ProductSpace<(AS1, AS2)>,
    >,
    OS1: Space,
    OS2: Space,
    AS1: Space,
    AS1::Element: Default,
    AS2: Space,
    AS2::Element: Clone,
{
    type State = E::State;
    type Observation = OS2::Element;
    type Action = AS2::Element;
    type ObservationSpace = OS2;
    type ActionSpace = AS2;
    type Pomdp = SecondPlayerView<E::Pomdp>;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        Ok(SecondPlayerView {
            inner: self.inner.build_pomdp()?,
        })
    }
}

impl<E, OS1, OS2, AS1, AS2> EnvStructure for SecondPlayerView<E>
where
    E: EnvStructure<
        ObservationSpace = ProductSpace<(OS1, OS2)>,
        ActionSpace = ProductSpace<(AS1, AS2)>,
    >,
    OS2: Space,
    AS2: Space,
{
    type ObservationSpace = OS2;
    type ActionSpace = AS2;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space().inner_spaces.1
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space().inner_spaces.1
    }
    fn reward_range(&self) -> (f64, f64) {
        self.inner.reward_range()
    }
    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<E, O1, O2, A1, A2> Pomdp for SecondPlayerView<E>
where
    E: Pomdp<Observation = (O1, O2), Action = (A1, A2)>,
    A1: Default,
    A2: Clone,
{
    type State = E::State;
    type Observation = O2;
    type Action = A2;

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        self.inner.initial_state(rng)
    }

    fn observe(&self, state: &Self::State, rng: &mut StdRng) -> Self::Observation {
        self.inner.observe(state, rng).1
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
        logger: &mut dyn Logger,
    ) -> (Option<Self::State>, f64, bool) {
        let joint_action = (Default::default(), action.clone());
        self.inner.step(state, &joint_action, rng, logger)
    }
}

impl<E, O1, O2, A1, A2> Environment for SecondPlayerView<E>
where
    E: Environment<Observation = (O1, O2), Action = (A1, A2)>,
    A1: Default,
    A2: Clone,
{
    type Observation = O2;
    type Action = A2;

    fn step(
        &mut self,
        action: &Self::Action,
        logger: &mut dyn Logger,
    ) -> (Option<Self::Observation>, f64, bool) {
        let joint_action = (Default::default(), action.clone());
        let (joint_observation, reward, episode_done) = self.inner.step(&joint_action, logger);
        let observation = joint_observation.map(|o| o.1);
        (observation, reward, episode_done)
    }

    fn reset(&mut self) -> Self::Observation {
        self.inner.reset().1
    }
}
