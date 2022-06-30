//! Fruit collection gridworlds.
use crate::envs::{CloneBuild, EnvStructure, Environment, Successor};
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::spaces::{
    ArraySpace, BooleanSpace, BoxSpace, FiniteSpace, IndexSpace, IndexedTypeSpace, IntervalSpace,
    PowerSpace, ProductSpace, Space, TupleSpace2,
};
use crate::utils::coord_vector::CoordVector;
use crate::Prng;
use enum_map::{enum_map, Enum, EnumMap};
use rand::distributions::Standard;
use rand::prelude::*;
use relearn_derive::Indexed;
use serde::{Deserialize, Serialize};
use slice_of_array::SliceFlatExt;
use std::fmt::{self, Display};

/// Cell contents
pub type Cell = Option<Fruit>;

/// Agent view of a cell's contents
///
/// The view is always relative to the target agent's position so no flag is needed for itself.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Indexed)]
pub enum CellView {
    Empty,
    Apple,
    Cherry,
    OtherAgent,
}

impl From<Cell> for CellView {
    fn from(cell: Cell) -> Self {
        match cell {
            None => Self::Empty,
            Some(Fruit::Apple) => Self::Apple,
            Some(Fruit::Cherry) => Self::Cherry,
        }
    }
}

impl Display for CellView {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Empty => ' ',
                Self::Apple => 'A',
                Self::Cherry => 'C',
                Self::OtherAgent => 'O',
            }
        )
    }
}

pub type GridVec = CoordVector<usize, 2>;

/// a - b % size
const fn wrapping_sub(a: GridVec, b: GridVec, size: GridVec) -> GridVec {
    let CoordVector([ai, aj]) = a;
    let CoordVector([bi, bj]) = b;
    let CoordVector([si, sj]) = size;
    CoordVector([(si + ai - bi) % si, (sj + aj - bj) % sj])
}

/// Generate a grid view relative to a given position
fn grid_view<const W: usize, const H: usize, const VW: usize, const VH: usize>(
    cells: &[[Cell; W]; H],
    pos: GridVec,
    other_agent_pos: GridVec,
) -> Box<[[CellView; VW]; VH]> {
    let mut view = Box::new([[CellView::Empty; VW]; VH]);

    // Top left corner of the viewport when `pos` is in the middle
    let rel = wrapping_sub(pos, CoordVector([VH / 2, VW / 2]), CoordVector([H, W]));
    for i in 0..VH {
        let cells_row = cells[(rel[0] + i) % H];
        for j in 0..VW {
            view[i][j] = cells_row[(rel[1] + j) % W].into();
        }
    }

    // Relative position of the other agent
    let other_rel_pos = wrapping_sub(other_agent_pos, rel, CoordVector([H, W]));
    if other_rel_pos[0] < VH && other_rel_pos[1] < VW {
        // Should always be empty because agents consume the contents of cells they enter
        assert_eq!(view[other_rel_pos[0]][other_rel_pos[1]], CellView::Empty);
        view[other_rel_pos[0]][other_rel_pos[1]] = CellView::OtherAgent;
    }

    view
}

/// Fruit types
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Enum)]
pub enum Fruit {
    Apple,
    Cherry,
}

impl Distribution<Fruit> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Fruit {
        if rng.gen() {
            Fruit::Apple
        } else {
            Fruit::Cherry
        }
    }
}

/// Player types
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Enum)]
enum Player {
    Principal,
    Assistant,
}

#[derive(Debug)]
pub struct FruitGameState<const W: usize, const H: usize> {
    cells: Box<[[Cell; W]; H]>,
    /// Player positions
    positions: EnumMap<Player, GridVec>,
    /// Goal fruit
    goal: Fruit,
    /// Number of remaining fruit of each type
    remaining: EnumMap<Fruit, usize>,
}

impl<const W: usize, const H: usize> FruitGameState<W, H> {
    fn observe<const VW: usize, const VH: usize>(
        &self,
    ) -> <JointObsSpace<VW, VH> as Space>::Element {
        let principal_view = grid_view(
            &self.cells,
            self.positions[Player::Principal],
            self.positions[Player::Assistant],
        );
        let CoordVector(principal_pos) = self.positions[Player::Principal];
        // If more than 2 fruit then change from BooleanSpace to IndexedTypeSpace
        let goal_is_apple = match self.goal {
            Fruit::Apple => true,
            Fruit::Cherry => false,
        };
        let principal_obs = PrincipalObs {
            visible_grid: principal_view,
            position: principal_pos,
            goal_is_apple,
        };

        let assistant_view = grid_view(
            &self.cells,
            self.positions[Player::Assistant],
            self.positions[Player::Principal],
        );
        let CoordVector(assistant_pos) = self.positions[Player::Assistant];
        let assistant_obs = AssistantObs {
            visible_grid: assistant_view,
            position: assistant_pos,
        };
        (principal_obs, assistant_obs)
    }

    fn step(&mut self, action: Move, player: Player) -> f64 {
        let pos = &mut self.positions[player];
        *pos = action.apply(*pos, CoordVector([H, W]));
        let cell = self.cells[pos[0]][pos[1]].take();
        match cell {
            None => 0.0,
            Some(fruit) => {
                self.remaining[fruit] -= 1;
                if fruit == self.goal {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    fn is_terminal(&self) -> bool {
        self.remaining.values().all(|&count| count == 0)
    }
}

/// Visible region of the grid centered on an agent
pub type VisibleGridSpace<const W: usize, const H: usize> =
    BoxSpace<PowerSpace<PowerSpace<IndexedTypeSpace<CellView>, W>, H>>;

/// Observation for the principal
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrincipalObs<const W: usize, const H: usize> {
    /// Visible part of the grid centered on own position
    pub visible_grid: Box<[[CellView; W]; H]>,
    /// Own position
    pub position: [usize; 2],
    /// Whether the goal is apple (true) or cherry (false).
    pub goal_is_apple: bool,
}

impl<const W: usize, const H: usize> Display for PrincipalObs<W, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let [i, j] = &self.position;
        writeln!(
            f,
            "[{}] ({} {})",
            if self.goal_is_apple { 'A' } else { 'C' },
            i,
            j
        )?;
        for row in self.visible_grid.iter() {
            for cell in row {
                write!(f, "{}", cell)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Observation space for the principal
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, ProductSpace, FiniteSpace)]
#[element(PrincipalObs<W, H>)]
pub struct PrincipalObsSpace<const W: usize, const H: usize> {
    pub visible_grid: VisibleGridSpace<W, H>,
    pub position: ArraySpace<IndexSpace, 2>,
    pub goal_is_apple: BooleanSpace,
}

/// Observation for the assistant
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssistantObs<const W: usize, const H: usize> {
    /// Visible part of the grid centered on own position
    pub visible_grid: Box<[[CellView; W]; H]>,
    /// Own position
    pub position: [usize; 2],
}

/// Observation space for the assistant
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, ProductSpace, FiniteSpace)]
#[element(AssistantObs<W, H>)]
pub struct AssistantObsSpace<const W: usize, const H: usize> {
    pub visible_grid: VisibleGridSpace<W, H>,
    pub position: ArraySpace<IndexSpace, 2>,
}

pub type JointObsSpace<const VW: usize, const VH: usize> =
    TupleSpace2<PrincipalObsSpace<VW, VH>, AssistantObsSpace<VW, VH>>;

/// Grid cell movement
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Indexed)]
pub enum Move {
    Still,
    Up,
    Down,
    Left,
    Right,
}

impl Default for Move {
    fn default() -> Self {
        Self::Still
    }
}

impl Move {
    /// Apply the move with wrapping around a grid of size `size`.
    const fn apply(self, pos: GridVec, size: GridVec) -> GridVec {
        let CoordVector([i, j]) = pos;
        let CoordVector([si, sj]) = size;
        match self {
            Move::Still => pos,
            Move::Up => CoordVector([(si - 1 + i) % si, j]),
            Move::Down => CoordVector([(i + 1) % si, j]),
            Move::Left => CoordVector([i, (sj - 1 + j) % sj]),
            Move::Right => CoordVector([i, (j + 1) % sj]),
        }
    }
}

/// Cooperative two-agent fruit collecting game.
///
/// Based on the paper "[Learning to Interactively Learn and Assist][l2ila]"
/// by Woodward et al. (2020)
///
/// [l2ila]: https://arxiv.org/abs/1906.10187
///
/// # Generic Parameters
/// * `W` - Grid width
/// * `H` - Grid height
/// * `VW` - Agent viewport width (centered on agent)
/// * `VH` - Agent viewport height (centered on agent)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FruitGame<const W: usize, const H: usize, const VW: usize, const VH: usize> {
    /// Number of fruit of each type
    pub num_fruit: usize,
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> FruitGame<W, H, VW, VH> {
    /// Initialize a new [`FruitGame`] instance.
    ///
    /// # Args
    /// * `num_fruit` - Number of fruit of each type.
    #[must_use]
    pub const fn new(num_fruit: usize) -> Self {
        Self { num_fruit }
    }
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> Default
    for FruitGame<W, H, VW, VH>
{
    fn default() -> Self {
        let target_inv_density = 2;
        let num_fruit_types = 2;
        let num_fruit = W * H / (target_inv_density * num_fruit_types);
        Self { num_fruit }
    }
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> CloneBuild
    for FruitGame<W, H, VW, VH>
{
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> EnvStructure
    for FruitGame<W, H, VW, VH>
{
    /// An observation for each agent
    type ObservationSpace = JointObsSpace<VW, VH>;
    /// An action for each agent
    type ActionSpace = TupleSpace2<IndexedTypeSpace<Move>, IndexedTypeSpace<Move>>;
    /// A reward for each agent
    type FeedbackSpace = TupleSpace2<IntervalSpace<Reward>, IntervalSpace<Reward>>;

    fn observation_space(&self) -> Self::ObservationSpace {
        let visible_grid = VisibleGridSpace::default(); // No dynamic structure
        let position = ArraySpace::new([IndexSpace::new(H), IndexSpace::new(W)]);
        let principal_obs_space = PrincipalObsSpace {
            visible_grid,
            position,
            goal_is_apple: BooleanSpace,
        };
        let assistant_obs_space = AssistantObsSpace {
            visible_grid,
            position,
        };
        TupleSpace2(principal_obs_space, assistant_obs_space)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Default::default()
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        // Give both agents the same reward
        let reward_range = IntervalSpace::new(Reward(-2.0), Reward(2.0));
        TupleSpace2(reward_range, reward_range)
    }

    fn discount_factor(&self) -> f64 {
        0.95
    }
}

impl<const W: usize, const H: usize, const VW: usize, const VH: usize> Environment
    for FruitGame<W, H, VW, VH>
{
    type State = FruitGameState<W, H>;
    type Observation = <JointObsSpace<VW, VH> as Space>::Element;
    type Action = (Move, Move);
    type Feedback = (Reward, Reward);

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        let mut cells = Box::new([[None; W]; H]);
        let cells_slice = cells.flat_mut();
        let num_cells = cells_slice.len();
        assert!(num_cells != 0, "game grid must be nonempty");

        // Add fruit then shuffle
        // Want to leave the center cell (origin) empty for the agents
        // so first leave the last cell empty then swap
        let prefix = &mut cells_slice[..num_cells - 1];
        prefix[0..self.num_fruit].fill(Some(Fruit::Apple));
        prefix[self.num_fruit..2 * self.num_fruit].fill(Some(Fruit::Cherry));
        prefix.shuffle(rng);

        let origin = CoordVector([H / 2, W / 2]);

        // Swap origin and the last cell. The last cell is None so only need to go one direction.
        *cells_slice.last_mut().unwrap() = cells_slice[origin[0] * W + origin[1]].take();

        FruitGameState {
            cells,
            positions: enum_map! {
                Player::Principal => origin,
                Player::Assistant => origin,
            },
            goal: rng.gen(),
            remaining: enum_map! {
                Fruit::Apple => self.num_fruit,
                Fruit::Cherry => self.num_fruit,
            },
        }
    }

    fn observe(&self, state: &Self::State, _rng: &mut Prng) -> Self::Observation {
        state.observe()
    }

    fn step(
        &self,
        mut state: Self::State,
        action: &Self::Action,
        _rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        let (principal_action, assistant_action) = *action;

        let reward_principal = state.step(principal_action, Player::Principal);
        let reward_assistant = state.step(assistant_action, Player::Assistant);
        let mut reward_logger = logger.with_scope("reward").group();
        reward_logger.log_scalar("principal", reward_principal);
        reward_logger.log_scalar("assistant", reward_assistant);
        let reward = Reward(reward_principal + reward_assistant);
        let successor = if state.is_terminal() {
            Successor::Terminate
        } else {
            Successor::Continue(state)
        };
        (successor, (reward, reward))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::testing;
    use super::*;

    #[test]
    fn run_default() {
        testing::check_structured_env(&FruitGame::<5, 5, 5, 5>::new(4), 1000, 0);
    }
}
