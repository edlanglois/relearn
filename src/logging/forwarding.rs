use super::{Event, Id, LogError, Loggable, ScopedLogger, TimeSeriesEventLogger, TimeSeriesLogger};
use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    Log {
        event: Event,
        id: Id,
        value: Loggable,
    },
    EndEvent(Event),
}

impl Message {
    /// Log this message to the given logger.
    pub fn log<L: TimeSeriesLogger + ?Sized>(self, logger: &mut L) -> Result<(), LogError> {
        match self {
            Message::Log { event, id, value } => logger.id_log(event, id, value)?,
            Message::EndEvent(event) => logger.end_event(event),
        }
        Ok(())
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
    fn id_log(&mut self, event: Event, id: Id, value: Loggable) -> Result<(), LogError> {
        self.sender.send(Message::Log { event, id, value }).unwrap();
        Ok(())
    }

    fn end_event(&mut self, event: Event) {
        self.sender.send(Message::EndEvent(event)).unwrap();
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
