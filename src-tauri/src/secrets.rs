//! API keys in the OS credential store (Windows Credential Manager, macOS Keychain).
//! Owner: secrets/storage worker (high risk).
//! Rules: a key is never written to settings, logs, session files, exports or
//! error messages. `Secret` hides its value in `Debug` and has no `Display`.
//!
//! One entry per (role, base URL): service `"morago"`, user
//! `"<role>:<normalized base_url>"`. Linux uses the kernel keyutils store
//! (development only: the keys go away at logout).

use keyring_core::{CredentialStore, Entry, Error as StoreError};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use zeroize::Zeroize;

const SERVICE: &str = "morago";

/// Which provider role the key is for. One key per (role, base URL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Asr,
    Translate,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Asr => "asr",
            Role::Translate => "translate",
        }
    }
}

/// An API key. Read the value only with `expose` at the request site.
/// The memory is overwritten on drop.
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

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

pub fn set_key(role: Role, base_url: &str, key: &str) -> anyhow::Result<()> {
    let key = key.trim();
    if key.is_empty() {
        anyhow::bail!("API 키가 비어 있습니다. 키를 입력하세요.");
    }
    entry(role, base_url)?
        .set_password(key)
        .map_err(|e| store_error("저장", &e))
}

/// `Ok(None)` when no key is stored (a local server can work without a key).
pub fn get_key(role: Role, base_url: &str) -> anyhow::Result<Option<Secret>> {
    match entry(role, base_url)?.get_password() {
        Ok(value) => Ok(Some(Secret::new(value))),
        Err(StoreError::NoEntry) => Ok(None),
        Err(e) => Err(store_error("읽기", &e)),
    }
}

pub fn delete_key(role: Role, base_url: &str) -> anyhow::Result<()> {
    match entry(role, base_url)?.delete_credential() {
        Ok(()) | Err(StoreError::NoEntry) => Ok(()),
        Err(e) => Err(store_error("삭제", &e)),
    }
}

/// Masked hint for the UI, for example "••••3f9a"; `None` when no key.
pub fn key_hint(role: Role, base_url: &str) -> anyhow::Result<Option<String>> {
    Ok(get_key(role, base_url)?.map(|s| mask(s.expose())))
}

fn mask(key: &str) -> String {
    let count = key.chars().count();
    if count <= 8 {
        return "••••".into();
    }
    let tail: String = key.chars().skip(count - 4).collect();
    format!("••••{tail}")
}

/// Trims spaces and one trailing slash; lowercases the scheme and the host
/// (user info, path and query keep their case); drops the default port
/// (`:443` for https, `:80` for http) so both forms find one entry.
fn normalize_base_url(base_url: &str) -> String {
    let url = base_url.trim();
    let url = url.strip_suffix('/').unwrap_or(url);
    let (scheme, rest) = match url.split_once("://") {
        Some((s, r)) => (Some(s.to_ascii_lowercase()), r),
        None => (None, url),
    };
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(auth_end);
    let authority = match authority.rsplit_once('@') {
        Some((user, host)) => format!("{user}@{}", host.to_ascii_lowercase()),
        None => authority.to_ascii_lowercase(),
    };
    let default_port = match scheme.as_deref() {
        Some("https") => Some(":443"),
        Some("http") => Some(":80"),
        _ => None,
    };
    let authority = match default_port.and_then(|p| authority.strip_suffix(p)) {
        Some(short) => short.to_string(),
        None => authority,
    };
    match scheme {
        Some(s) => format!("{s}://{authority}{tail}"),
        None => format!("{authority}{tail}"),
    }
}

fn user_name(role: Role, base_url: &str) -> String {
    format!("{}:{}", role.as_str(), normalize_base_url(base_url))
}

fn entry(role: Role, base_url: &str) -> anyhow::Result<Entry> {
    let store = store()?;
    store
        .build(SERVICE, &user_name(role, base_url), None)
        .map_err(|e| store_error("열기", &e))
}

/// The OS store, made once. The error text is built here and has no key.
fn store() -> anyhow::Result<&'static Arc<CredentialStore>> {
    static STORE: OnceLock<Result<Arc<CredentialStore>, String>> = OnceLock::new();
    STORE
        .get_or_init(|| open_store().map_err(|e| describe(&e)))
        .as_ref()
        .map_err(|reason| anyhow::anyhow!("OS 자격 증명 저장소를 열 수 없습니다: {reason}"))
}

#[cfg(target_os = "windows")]
fn open_store() -> keyring_core::Result<Arc<CredentialStore>> {
    Ok(windows_native_keyring_store::Store::new()?)
}

#[cfg(target_os = "macos")]
fn open_store() -> keyring_core::Result<Arc<CredentialStore>> {
    Ok(apple_native_keyring_store::keychain::Store::new()?)
}

#[cfg(target_os = "linux")]
fn open_store() -> keyring_core::Result<Arc<CredentialStore>> {
    Ok(linux_keyutils_keyring_store::Store::new()?)
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn open_store() -> keyring_core::Result<Arc<CredentialStore>> {
    Err(StoreError::NotSupportedByStore(
        "이 OS는 지원하지 않습니다".into(),
    ))
}

fn store_error(action: &str, e: &StoreError) -> anyhow::Error {
    anyhow::anyhow!(
        "API 키 {action}에 실패했습니다({}). OS 자격 증명 저장소 권한을 확인하세요.",
        describe(e)
    )
}

/// A short reason without data from the store. `BadEncoding` and
/// `BadDataFormat` carry the stored bytes, so this never formats them.
fn describe(e: &StoreError) -> String {
    match e {
        StoreError::PlatformFailure(err) => format!("OS 오류: {err}"),
        StoreError::NoStorageAccess(err) => format!("저장소 접근 거부: {err}"),
        StoreError::NoEntry => "키 없음".into(),
        StoreError::BadEncoding(_) | StoreError::BadDataFormat(..) => {
            "저장된 키의 형식이 잘못됨".into()
        }
        StoreError::BadStoreFormat(_) => "저장소 형식 오류".into(),
        StoreError::TooLong(name, len) => format!("{name} 값이 {len}자 제한보다 김"),
        StoreError::Invalid(name, _) => format!("{name} 값이 잘못됨"),
        StoreError::Ambiguous(items) => format!("같은 항목이 {}개", items.len()),
        StoreError::NoDefaultStore => "저장소 없음".into(),
        StoreError::NotSupportedByStore(issue) => format!("지원하지 않음: {issue}"),
        _ => "알 수 없는 오류".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_shows_last_four_of_long_keys_only() {
        assert_eq!(mask("sk-abcdef123f9a"), "••••3f9a");
        assert_eq!(mask("123456789"), "••••6789");
        assert_eq!(mask("12345678"), "••••");
        assert_eq!(mask("abc"), "••••");
        assert_eq!(mask("키키키키키키키키키가나다라"), "••••가나다라");
    }

    #[test]
    fn normalize_trims_and_lowercases_scheme_and_host() {
        let n = normalize_base_url;
        assert_eq!(
            n("  HTTPS://API.OpenAI.com/v1/ "),
            "https://api.openai.com/v1"
        );
        assert_eq!(n("http://LocalHost:11434/V1"), "http://localhost:11434/V1");
        assert_eq!(n("https://host/v1//"), "https://host/v1/");
        assert_eq!(n("https://Ab:Cd@HOST/x"), "https://Ab:Cd@host/x");
        assert_eq!(n("HOST:8080/Path/"), "host:8080/Path");
        assert_eq!(n("https://HOST?Q=A"), "https://host?Q=A");
        assert_eq!(n("https://host:443/v1"), "https://host/v1");
        assert_eq!(n("http://host:80/v1/"), "http://host/v1");
        assert_eq!(n("http://host:8080"), "http://host:8080");
        assert_eq!(n("https://host:80/v1"), "https://host:80/v1");
        assert_eq!(n("http://host:8443"), "http://host:8443");
    }

    #[test]
    fn user_name_has_role_and_normalized_url() {
        assert_eq!(
            user_name(Role::Translate, "HTTPS://Api.X.com/v1/"),
            "translate:https://api.x.com/v1"
        );
        assert_eq!(user_name(Role::Asr, "http://h/"), "asr:http://h");
    }

    #[test]
    fn secret_debug_is_redacted() {
        let s = Secret::new("sk-very-secret-value".into());
        let text = format!("{s:?} {:?}", Some(&s));
        assert!(!text.contains("very-secret"), "{text}");
        assert_eq!(format!("{s:?}"), "Secret(***)");
    }

    #[test]
    fn empty_key_is_rejected_before_the_store() {
        let err = set_key(Role::Asr, "http://empty.invalid", "   ").unwrap_err();
        assert!(err.to_string().contains("비어"));
    }

    #[test]
    fn store_errors_never_show_stored_bytes() {
        let bytes = b"sk-leak-me".to_vec();
        for e in [
            StoreError::BadEncoding(bytes.clone()),
            StoreError::BadDataFormat(bytes.clone(), "sk-leak-me".into()),
        ] {
            let text = format!("{:#}", store_error("읽기", &e));
            assert!(!text.contains("leak"), "{text}");
        }
    }

    /// Real OS store. Run with `cargo test secrets -- --ignored` (CI does
    /// this on Windows and macOS).
    #[test]
    #[ignore = "needs the OS credential store"]
    fn os_store_round_trip() {
        let id = uuid::Uuid::new_v4();
        let url = format!("https://RoundTrip.Test/v1/{id}/");
        let key = "sk-test-round-trip-1234abcd";
        delete_key(Role::Translate, &url).unwrap();
        assert!(get_key(Role::Translate, &url).unwrap().is_none());
        assert!(key_hint(Role::Translate, &url).unwrap().is_none());

        set_key(Role::Translate, &url, key).unwrap();
        // The same entry by a differently written URL.
        let same = format!("  HTTPS://ROUNDTRIP.TEST/v1/{id}  ");
        assert_eq!(
            get_key(Role::Translate, &same).unwrap().unwrap().expose(),
            key
        );
        assert_eq!(
            key_hint(Role::Translate, &url).unwrap().as_deref(),
            Some("••••abcd")
        );
        // Other role: other entry.
        assert!(get_key(Role::Asr, &url).unwrap().is_none());

        set_key(Role::Translate, &url, " sk-second-key-9999 \n").unwrap();
        assert_eq!(
            get_key(Role::Translate, &url).unwrap().unwrap().expose(),
            "sk-second-key-9999"
        );

        delete_key(Role::Translate, &url).unwrap();
        assert!(get_key(Role::Translate, &url).unwrap().is_none());
        delete_key(Role::Translate, &url).unwrap();
    }
}
