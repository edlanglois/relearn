//! Anonymous Cartesian product spaces.
use super::ProductSpace;

/// Cartesian product of two spaces; elements are tuples
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, ProductSpace)]
pub struct TupleSpace2<A, B>(pub A, pub B);

/// Cartesian product of three spaces; elements are tuples
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, ProductSpace)]
pub struct TupleSpace3<A, B, C>(pub A, pub B, pub C);

/// Cartesian product of four spaces; elements are tuples
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, ProductSpace)]
pub struct TupleSpace4<A, B, C, D>(pub A, pub B, pub C, pub D);

/// Cartesian product of five spaces; elements are tuples
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, ProductSpace)]
pub struct TupleSpace5<A, B, C, D, E>(pub A, pub B, pub C, pub D, pub E);
