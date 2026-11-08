//! API keys in the OS credential store (Windows Credential Manager, macOS Keychain).
//! Owner: secrets/storage worker (high risk).
//! Rules: a key is never written to settings, logs, session files, exports or
//! error messages. `Secret` hides its value in `Debug`/`Display`.

use serde::{Deserialize, Serialize};

/// Which provider role the key is for. One key per (role, base URL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Asr,
    Translate,
}

/// An API key. Read the value only with `expose` at the request site.
pub struct Secret(String);

impl Secret {
    pub fn new(value: String) -> Self {
        Self(value)
    }
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(***)")
    }
}

pub fn set_key(role: Role, base_url: &str, key: &str) -> anyhow::Result<()> {
    let _ = (role, base_url, key);
    anyhow::bail!("key storage is not implemented yet")
}

/// `Ok(None)` when no key is stored (a local server can work without a key).
pub fn get_key(role: Role, base_url: &str) -> anyhow::Result<Option<Secret>> {
    let _ = (role, base_url);
    Ok(None)
}

pub fn delete_key(role: Role, base_url: &str) -> anyhow::Result<()> {
    let _ = (role, base_url);
    Ok(())
}

/// Masked hint for the UI, for example "••••3f9a"; `None` when no key.
pub fn key_hint(role: Role, base_url: &str) -> anyhow::Result<Option<String>> {
    let _ = (role, base_url);
    Ok(None)
}
