//! Command-line logger
use super::{
    Event, Id, IncompatibleValueError, LogError, Loggable, ScopedLogger, TimeSeriesEventLogger,
    TimeSeriesLogger,
};
use enum_map::{enum_map, EnumMap};
use std::borrow::Cow;
use std::collections::{btree_map, BTreeMap};
use std::convert::TryInto;
use std::fmt;
use std::ops::Drop;
use std::time::{Duration, Instant};

/// Time series logger that writes summaries to stderr.
#[derive(Debug, Clone)]
pub struct CLILogger {
    events: EnumMap<Event, EventLog>,

    display_period: Duration,
    // Tries to avoid logging at the end of high-frequency events,
    // but will log anyways if urgent_display_period is reached.
    urgent_display_period: Duration,
    last_display_time: Instant,

    average_between_displays: bool,
}

impl CLILogger {
    pub fn new(display_period: Duration, average_between_displays: bool) -> Self {
        let urgent_display_period = display_period.mul_f32(1.1);
        Self {
            events: enum_map! { _ => EventLog::new() },
            display_period,
            urgent_display_period: urgent_display_period,
            last_display_time: Instant::now(),
            average_between_displays,
        }
    }

    /// Display the summary and clear all stored data.
    pub fn display(&mut self) {
        println!();
        for (event, mut event_log) in self.events.iter_mut() {
            let summary_size = event_log.index - event_log.summary_start_index;
            if summary_size == 0 {
                continue;
            }

            print!("==== ");
            if self.average_between_displays {
                print!(
                    "{:?}s {} - {}",
                    event,
                    event_log.summary_start_index,
                    event_log.index - 1
                );
            } else {
                print!("{:?} {}", event, event_log.index - 1);
            }
            print!(
                " ({:?} / event)",
                event_log.summary_duration / summary_size.try_into().unwrap()
            );
            println!(" ====");

            for (id, aggregator) in &mut event_log.aggregators {
                println!("{}: {}", id, aggregator);
                aggregator.clear();
            }
            event_log.summary_start_index = event_log.index;
        }
        self.last_display_time = Instant::now();
    }
}

impl TimeSeriesLogger for CLILogger {
    fn id_log<'a>(&mut self, event: Event, id: Id, value: Loggable) -> Result<(), LogError> {
        let aggregators = &mut self.events[event].aggregators;
        match aggregators.entry(id) {
            btree_map::Entry::Vacant(e) => {
                e.insert(Aggregator::new(value));
            }
            btree_map::Entry::Occupied(mut e) => {
                if let Err((value, expected)) = e.get_mut().set_pending(value) {
                    // TODO Try to get the original id back instead of copying.
                    // Maybe raw_entry will allow it once stable?
                    return Err(IncompatibleValueError {
                        id: e.key().clone(),
                        value,
                        expected,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    fn end_event(&mut self, event: Event) {
        let event_info = &mut self.events[event];
        event_info.index += 1;

        for aggregator in event_info.aggregators.values_mut() {
            aggregator.commit()
        }

        let time_since_display = self.last_display_time.elapsed();
        event_info.summary_duration = time_since_display;
        if time_since_display < self.display_period {
            return;
        }

        // Don't display at the end of a step unless the urgent_display_period is reached
        if (event == Event::EnvStep
            || event == Event::AgentPolicyOptStep
            || event == Event::AgentValueOptStep)
            && time_since_display < self.urgent_display_period
        {
            return;
        }

        self.display();
    }

    fn event_logger(&mut self, event: Event) -> TimeSeriesEventLogger {
        TimeSeriesEventLogger {
            logger: self,
            event,
        }
    }

    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn TimeSeriesLogger> {
        ScopedLogger::new(self, scope)
    }
}

impl Drop for CLILogger {
    fn drop(&mut self) {
        // Ensure everything is flushed.
        self.display();
    }
}

#[derive(Debug, Clone, PartialEq)]
struct EventLog {
    /// Global index for this event
    index: u64,
    /// Value of `index` at the start of this summary period
    summary_start_index: u64,
    /// Duration of this summary period to the most recent update
    summary_duration: Duration,
    /// An aggregator for each log entry.
    aggregators: BTreeMap<Id, Aggregator>,
}

impl EventLog {
    #[allow(clippy::missing_const_for_fn)] // Duration & BTreeMap const new not stabilized
    pub fn new() -> Self {
        Self {
            index: 0,
            summary_start_index: 0,
            summary_duration: Duration::new(0, 0),
            aggregators: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Aggregator {
    Nothing(WithPending<NothingAccumulator>),
    ScalarMean(WithPending<MeanAccumulator>),
    IndexDistribution(WithPending<IndexDistributionAccumulator>),
    MessageCounts(WithPending<MessageAccumulator>),
}

impl Aggregator {
    // future-proofing in case loggable ends up containing non-copy values
    #[allow(clippy::needless_pass_by_value)]
    /// Create a new aggregator from a logged value value.
    fn new(value: Loggable) -> Self {
        use Aggregator::*;
        match value {
            Loggable::Nothing => Nothing(WithPending::new(NothingAccumulator::new())),
            Loggable::Scalar(x) => ScalarMean(WithPending {
                accumulator: MeanAccumulator::new(),
                pending: Some(x),
            }),
            Loggable::IndexSample { value, size } => IndexDistribution(WithPending {
                accumulator: IndexDistributionAccumulator::new(size),
                pending: Some(value),
            }),
            Loggable::Message(message) => MessageCounts(WithPending {
                accumulator: MessageAccumulator::new(),
                pending: Some(message),
            }),
        }
    }

    /// Set the pending value.
    ///
    /// Returns Err((value, expected)) if the value is incompatible with this aggregator.
    fn set_pending(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        use Aggregator::*;
        match self {
            Nothing(a) => a.set_pending(value),
            ScalarMean(a) => a.set_pending(value),
            IndexDistribution(a) => a.set_pending(value),
            MessageCounts(a) => a.set_pending(value),
        }
    }

    /// Commit the pending value to the accumulator.
    fn commit(&mut self) {
        use Aggregator::*;
        match self {
            Nothing(a) => a.commit(),
            ScalarMean(a) => a.commit(),
            IndexDistribution(a) => a.commit(),
            MessageCounts(a) => a.commit(),
        }
    }

    /// Clear the accumulated value.
    fn clear(&mut self) {
        use Aggregator::*;
        match self {
            Nothing(a) => a.clear(),
            ScalarMean(a) => a.clear(),
            IndexDistribution(a) => a.clear(),
            MessageCounts(a) => a.clear(),
        }
    }
}

impl fmt::Display for Aggregator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Aggregator::*;
        match self {
            Nothing(a) => a.fmt(f),
            ScalarMean(a) => a.fmt(f),
            IndexDistribution(a) => a.fmt(f),
            MessageCounts(a) => a.fmt(f),
        }
    }
}

struct WithPending<A: Accumulator> {
    accumulator: A,
    pending: Option<<A as Accumulator>::Prepared>,
}

// derive(...) does not work because of the associated type

impl<A> fmt::Debug for WithPending<A>
where
    A: Accumulator + fmt::Debug,
    <A as Accumulator>::Prepared: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WithPending")
            .field("accumulator", &self.accumulator)
            .field("pending", &self.pending)
            .finish()
    }
}

impl<A> Clone for WithPending<A>
where
    A: Accumulator + Clone,
    <A as Accumulator>::Prepared: Clone,
{
    fn clone(&self) -> Self {
        Self {
            accumulator: self.accumulator.clone(),
            pending: self.pending.clone(),
        }
    }
}

impl<A> Copy for WithPending<A>
where
    A: Accumulator + Copy,
    <A as Accumulator>::Prepared: Copy,
{
}

impl<A> PartialEq for WithPending<A>
where
    A: Accumulator + PartialEq,
    <A as Accumulator>::Prepared: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.accumulator == other.accumulator && self.pending == other.pending
    }
}

impl<A> Eq for WithPending<A>
where
    A: Accumulator + Eq,
    <A as Accumulator>::Prepared: Eq,
{
}

impl<A: Accumulator> WithPending<A> {
    pub fn new(accumulator: A) -> Self {
        Self {
            accumulator,
            pending: None,
        }
    }

    fn set_pending(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        self.pending = Some(self.accumulator.prepare(value)?);
        Ok(())
    }

    fn commit(&mut self) {
        if let Some(value) = self.pending.take() {
            self.accumulator.insert(value);
        }
    }

    fn clear(&mut self) {
        self.accumulator.clear();
    }
}

impl<A: Accumulator> fmt::Display for WithPending<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.accumulator.fmt(f)
    }
}

/// Accumulate statistics of a loggable.
trait Accumulator: 'static + fmt::Display {
    /// Type for prepared values.
    type Prepared;

    /// Prepare a value to be inserted into the accumulator.
    ///
    /// Used when a value is logged during an event.
    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)>;

    /// Insert a new prepared value into the accumulation.
    ///
    /// Used at the end of an event.
    fn insert(&mut self, value: Self::Prepared);

    /// Clear the accumulated values.
    fn clear(&mut self);
}

/// Accumulates nothing
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct NothingAccumulator;

impl NothingAccumulator {
    pub const fn new() -> Self {
        Self
    }
}

impl Accumulator for NothingAccumulator {
    type Prepared = ();

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        if let Loggable::Nothing = value {
            Ok(())
        } else {
            Err((value, "Nothing".into()))
        }
    }

    fn insert(&mut self, _: Self::Prepared) {}

    fn clear(&mut self) {}
}

impl fmt::Display for NothingAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Nothing")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MeanAccumulator {
    sum: f64,
    count: u64,
}

impl MeanAccumulator {
    pub const fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl Accumulator for MeanAccumulator {
    type Prepared = f64;

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        if let Loggable::Scalar(x) = value {
            Ok(x)
        } else {
            Err((value, "Scalar".into()))
        }
    }

    fn insert(&mut self, value: Self::Prepared) {
        self.sum += value;
        self.count += 1;
    }

    fn clear(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl fmt::Display for MeanAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.sum / (self.count as f64))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexDistributionAccumulator {
    counts: Vec<u64>,
}

impl IndexDistributionAccumulator {
    fn new(size: usize) -> Self {
        Self {
            counts: vec![0; size],
        }
    }
}

impl Accumulator for IndexDistributionAccumulator {
    type Prepared = usize;

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        match value {
            Loggable::IndexSample { value, size } if self.counts.len() == size => Ok(value),
            v => Err((v, format!("IndexSample{{size: {}}}", self.counts.len()))),
        }
    }

    fn insert(&mut self, value: Self::Prepared) {
        self.counts[value] += 1;
    }

    fn clear(&mut self) {
        for count in &mut self.counts {
            *count = 0;
        }
    }
}

impl fmt::Display for IndexDistributionAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let total: u64 = self.counts.iter().sum();
        if total > 0 {
            write!(f, "[")?;
            let mut first = true;
            for c in &self.counts {
                if first {
                    first = false;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{:.3}", (*c as f64) / (total as f64))?;
            }
            write!(f, "]")
        } else {
            write!(f, "None")
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MessageAccumulator {
    /// Number of occurrences of each unique message.
    message_counts: BTreeMap<Cow<'static, str>, usize>,
}

impl MessageAccumulator {
    #[allow(clippy::missing_const_for_fn)] // BTreeMap const new not stabilized
    pub fn new() -> Self {
        Self {
            message_counts: BTreeMap::new(),
        }
    }
}

impl Accumulator for MessageAccumulator {
    type Prepared = Cow<'static, str>;

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        if let Loggable::Message(message) = value {
            Ok(message)
        } else {
            Err((value, "Message".into()))
        }
    }

    fn insert(&mut self, message: Self::Prepared) {
        *self.message_counts.entry(message).or_insert(0) += 1;
    }

    fn clear(&mut self) {
        self.message_counts.clear();
    }
}

impl fmt::Display for MessageAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.message_counts.len() == 1 {
            for (message, count) in &self.message_counts {
                write!(f, "[x{}] {}", count, message)?;
            }
        } else {
            for (message, count) in &self.message_counts {
                write!(f, "\n\t[x{}] {}", count, message)?;
            }
        }
        Ok(())
    }
}
