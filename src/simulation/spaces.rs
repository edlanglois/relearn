//! Common space bounds for simulation.
use crate::logging::Loggable;
use crate::spaces::{ElementRefInto, SampleSpace, Space};
use std::fmt::Debug;

pub trait CommonBaseSpace: Space + ElementRefInto<Loggable> + Debug {}
impl<T: Space + ElementRefInto<Loggable> + Debug> CommonBaseSpace for T {}

pub trait CommonObservationSpace: CommonBaseSpace {}
impl<T: CommonBaseSpace> CommonObservationSpace for T {}

pub trait CommonActionSpace: CommonBaseSpace + SampleSpace {}
impl<T: CommonBaseSpace + SampleSpace> CommonActionSpace for T {}
