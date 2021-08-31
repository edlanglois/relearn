use super::{Event, Id, LogError, Loggable, ScopedLogger, TimeSeriesEventLogger, TimeSeriesLogger};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    Log {
        event: Event,
        id: Id,
        value: Loggable,
    },
    StartEvent {
        event: Event,
        time: Instant,
    },
    EndEvent {
        event: Event,
        time: Instant,
    },
}

impl Message {
    /// Log this message to the given logger.
    pub fn log<L: TimeSeriesLogger + ?Sized>(self, logger: &mut L) -> Result<(), LogError> {
        match self {
            Message::Log { event, id, value } => logger.log_(event, id, value),
            Message::StartEvent { event, time } => logger.start_event_(event, time),
            Message::EndEvent { event, time } => logger.end_event_(event, time),
        }
    }
}

/// Logger that forwards messages through a channel.
#[derive(Debug, Clone)]
pub struct ForwardingLogger {
    sender: Sender<Message>,
}

impl ForwardingLogger {
    /// Initialize a new logger and channel.
    pub fn new() -> (Self, Receiver<Message>) {
        let (sender, receiver) = mpsc::channel();
        (Self::with_sender(sender), receiver)
    }

    pub fn with_sender(sender: Sender<Message>) -> Self {
        Self { sender }
    }
}

impl TimeSeriesLogger for ForwardingLogger {
    fn log_(&mut self, event: Event, id: Id, value: Loggable) -> Result<(), LogError> {
        self.sender
            .send(Message::Log { event, id, value })
            .map_err(|e| LogError::InternalError(Box::new(e)))
    }

    fn start_event_(&mut self, event: Event, time: Instant) -> Result<(), LogError> {
        self.sender
            .send(Message::StartEvent { event, time })
            .map_err(|e| LogError::InternalError(Box::new(e)))
    }

    fn end_event_(&mut self, event: Event, time: Instant) -> Result<(), LogError> {
        self.sender
            .send(Message::EndEvent { event, time })
            .map_err(|e| LogError::InternalError(Box::new(e)))
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

/// Loop receiving and logging messages from a channel.
///
/// Ends once the channel closes or if any log produces an error.
pub fn log_receive_loop<L: TimeSeriesLogger + ?Sized>(
    receiver: &mut Receiver<Message>,
    logger: &mut L,
) -> Result<(), LogError> {
    while let Ok(message) = receiver.recv() {
        message.log(logger)?;
    }
    Ok(())
}
