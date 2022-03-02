use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// Serialize to file(s) or load from serialized file(s).
pub trait SaveLoad {
    type SaveErr;
    type LoadErr;

    /// Serialize to file(s).
    ///
    /// May create auxiliary files with names created by appending extensions to the given path.
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Self::SaveErr>;

    /// Load from file(s).
    ///
    /// May require additional auxiliary files created by `SaveLoad::Save`.
    fn load<P: AsRef<Path>>(path: P) -> Result<Self, Self::LoadErr>
    where
        Self: Sized;
}

#[derive(Debug, Error)]
pub enum SerdeSaveLoadError {
    #[error("file error {0}")]
    Io(#[from] std::io::Error),
    #[error("(de)serialization error {0}")]
    Serialize(#[from] serde_cbor::Error),
}

impl<T: Serialize + DeserializeOwned> SaveLoad for T {
    type SaveErr = SerdeSaveLoadError;
    type LoadErr = SerdeSaveLoadError;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Self::SaveErr> {
        let file = File::create(path)?;
        serde_cbor::to_writer(file, self)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(path: P) -> Result<Self, Self::LoadErr>
    where
        Self: Sized,
    {
        let file = File::open(path)?;
        let this = serde_cbor::from_reader(file)?;
        Ok(this)
    }
}
