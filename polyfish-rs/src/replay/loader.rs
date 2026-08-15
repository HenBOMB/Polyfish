use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::{Replay, ReplayError, validate_replay};

pub fn load_replay(path: impl AsRef<Path>) -> Result<Replay, ReplayError> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| ReplayError::Io {
        file: path.to_path_buf(),
        source,
    })?;
    load_replay_reader(BufReader::new(file), Some(path))
}

pub fn load_replay_reader(reader: impl Read, path: Option<&Path>) -> Result<Replay, ReplayError> {
    let file = path.unwrap_or_else(|| Path::new("<reader>"));
    let replay: Replay = serde_json::from_reader(reader).map_err(|source| ReplayError::Json {
        file: file.to_path_buf(),
        source,
    })?;
    validate_replay(&replay, path)?;
    Ok(replay)
}

pub fn save_replay(replay: &Replay, path: impl AsRef<Path>) -> Result<(), ReplayError> {
    let path = path.as_ref();
    validate_replay(replay, Some(path))?;
    let file = File::create(path).map_err(|source| ReplayError::Io {
        file: path.to_path_buf(),
        source,
    })?;
    serde_json::to_writer_pretty(file, replay).map_err(|source| ReplayError::Json {
        file: path.to_path_buf(),
        source,
    })
}
