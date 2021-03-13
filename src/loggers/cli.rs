use super::{Event, Logger};
use enum_map::{enum_map, EnumMap};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Logger that writes summaries to stderr.
pub struct CLILogger {
    events: EnumMap<Event, EventLog>,

    display_period: Duration,
    last_display_time: Instant,

    average_between_displays: bool,
}

impl CLILogger {
    pub fn new(display_period: Duration, average_between_displays: bool) -> Self {
        CLILogger {
            events: enum_map! { _ => EventLog::new() },
            display_period,
            last_display_time: Instant::now(),
            average_between_displays,
        }
    }

    /// Display the summary and clear all stored data.
    pub fn display(&mut self) {
        println!("");
        for (event, mut event_log) in self.events.iter_mut() {
            let summary_size = (event_log.index - event_log.summary_start_index) as i64 - 1;
            if summary_size <= 0 {
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
                event_log.summary_duration / (summary_size as u32)
            );
            println!(" ====");

            for (name, value) in event_log.scalars.iter() {
                println!("{}: {}", name, value);
            }
            event_log.summary_start_index = event_log.index;
        }
        self.last_display_time = Instant::now();
    }
}

impl Logger for CLILogger {
    fn log_scalar(&mut self, event: Event, name: &'static str, value: f64) {
        self.events[event].pending_scalars.insert(name, value);
    }

    fn done(&mut self, event: Event) {
        let event_info = &mut self.events[event];
        event_info.index += 1;

        let summary_size = event_info.index - event_info.summary_start_index;
        if !self.average_between_displays || summary_size == 1 {
            std::mem::swap(&mut event_info.scalars, &mut event_info.pending_scalars);
        } else {
            let weight = (summary_size as f64).recip();
            for (name, new_value) in event_info.pending_scalars.iter() {
                let summary_value = match event_info.scalars.get_mut(name) {
                    Some(value) => value,
                    None => panic!("Value {} missing from previous {:?} iteration", name, event),
                };
                *summary_value *= 1.0 - weight;
                *summary_value += weight * new_value;
            }
        }
        event_info.pending_scalars.clear();

        let time_since_display = self.last_display_time.elapsed();
        event_info.summary_duration = time_since_display;
        if time_since_display < self.display_period {
            return;
        }

        // Don't output after steps - prefer complete episodes or epochs
        if let Event::Step = event {
            return;
        }

        self.display();
    }

    fn close(&mut self) {
        self.display();
    }
}

struct EventLog {
    /// Global index for this event
    index: u64,
    /// Value of `index` at the start of this summary period
    summary_start_index: u64,
    /// Duration of this summary period to the most recent update
    summary_duration: Duration,
    /// Scalars associated with completed events.
    scalars: BTreeMap<&'static str, f64>,
    /// Pending scalars for the current incomplete event.
    pending_scalars: BTreeMap<&'static str, f64>,
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            index: 0,
            summary_start_index: 0,
            summary_duration: Duration::new(0, 0),
            scalars: BTreeMap::new(),
            pending_scalars: BTreeMap::new(),
        }
    }
}
