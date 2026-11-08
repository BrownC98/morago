//! OpenAI-compatible chat completions provider.

use super::{History, Translator, clean_output, system_prompt};
use crate::secrets::Secret;
use crate::types::Endpoint;
use anyhow::{Context, anyhow};
use serde_json::{Value, json};
use std::time::Duration;

/// Longest server error text put into a message.
const MAX_ERROR_CHARS: usize = 300;

pub(crate) struct CustomTranslator {
    endpoint: Endpoint,
    url: String,
    key: Option<Secret>,
    client: reqwest::blocking::Client,
    history: History,
}

impl CustomTranslator {
    pub(crate) fn new(endpoint: Endpoint, key: Option<Secret>) -> anyhow::Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(60))
            .build()
            .context("HTTP 클라이언트를 만들지 못했습니다")?;
        let url = format!(
            "{}/chat/completions",
            endpoint.base_url.trim_end_matches('/')
        );
        Ok(Self {
            endpoint,
            url,
            key,
            client,
            history: History::default(),
        })
    }

    fn request_body(&self, text: &str, source: Option<&str>, target: &str) -> Value {
        json!({
            "model": self.endpoint.model,
            "temperature": 0.2,
            "stream": false,
            "messages": [
                {"role": "system", "content": system_prompt(source, target, &self.history)},
                {"role": "user", "content": text},
            ],
        })
    }

    /// Removes the key from text that comes from the server or the network stack.
    fn redact(&self, text: &str) -> String {
        match &self.key {
            Some(key) if !key.expose().is_empty() => text.replace(key.expose(), "***"),
            _ => text.to_string(),
        }
    }
}

/// The server's error text: `error.message`, `error` or the raw body.
fn error_text(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let text = parsed
        .as_ref()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("error"))
                .or_else(|| v.get("message"))
        })
        .and_then(Value::as_str)
        .unwrap_or(body)
        .trim();
    text.to_string()
}

/// Cuts user-facing server text. Call it after `redact`: a key cut in half
/// would no longer match and its prefix would stay (review of ce83816).
fn shorten(text: &str) -> String {
    text.chars().take(MAX_ERROR_CHARS).collect()
}

/// `choices[0].message.content` of a chat completions response.
fn parse_content(body: &str) -> anyhow::Result<String> {
    let value: Value =
        serde_json::from_str(body).map_err(|_| anyhow!("번역 서버의 응답이 JSON이 아닙니다"))?;
    value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("번역 서버의 응답에 번역문이 없습니다"))
}

impl Translator for CustomTranslator {
    fn translate(
        &mut self,
        text: &str,
        source_language: Option<&str>,
        target_language: &str,
    ) -> anyhow::Result<String> {
        let body = self.request_body(text, source_language, target_language);
        let mut request = self.client.post(&self.url).json(&body);
        if let Some(key) = &self.key {
            request = request.bearer_auth(key.expose());
        }
        let response = request.send().map_err(|e| {
            anyhow!(
                "번역 서버에 연결하지 못했습니다: {}",
                self.redact(&e.without_url().to_string())
            )
        })?;
        let status = response.status();
        let response_text = response.text().map_err(|e| {
            anyhow!(
                "번역 서버의 응답을 읽지 못했습니다: {}",
                self.redact(&e.without_url().to_string())
            )
        })?;
        if !status.is_success() {
            return Err(anyhow!(
                "번역 서버 오류 (HTTP {}): {}",
                status.as_u16(),
                shorten(&self.redact(&error_text(&response_text)))
            ));
        }
        let translation = clean_output(&parse_content(&response_text)?, text);
        if translation.is_empty() {
            return Err(anyhow!("번역 서버가 빈 번역문을 돌려주었습니다"));
        }
        self.history.push(text);
        Ok(translation)
    }

    fn label(&self) -> String {
        format!("사용자 지정: {}", self.endpoint.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    /// One-shot HTTP server: answers one request, returns the raw request.
    fn mock_server(status: &str, body: &str) -> (String, JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let reply = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut data = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).unwrap();
                data.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&data).to_string();
                if let Some(split) = text.find("\r\n\r\n") {
                    let length = text[..split]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if data.len() >= split + 4 + length {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            stream.write_all(reply.as_bytes()).unwrap();
            String::from_utf8_lossy(&data).to_string()
        });
        (base_url, handle)
    }

    fn translator(base_url: String, key: Option<&str>) -> CustomTranslator {
        let endpoint = Endpoint {
            base_url,
            model: "gpt-test".into(),
        };
        CustomTranslator::new(endpoint, key.map(|k| Secret::new(k.into()))).unwrap()
    }

    fn ok_body(content: &str) -> String {
        json!({"choices": [{"message": {"role": "assistant", "content": content}}]}).to_string()
    }

    fn request_json(raw: &str) -> Value {
        serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap()).unwrap()
    }

    #[test]
    fn request_shape_and_bearer_with_key() {
        let (base_url, server) = mock_server("200 OK", &ok_body("Translation: \"안녕하세요\""));
        let mut t = translator(format!("{base_url}/"), Some("sk-secret-123"));
        let out = t.translate("Hello", Some("en"), "ko").unwrap();
        assert_eq!(out, "안녕하세요");
        let raw = server.join().unwrap();
        assert!(
            raw.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "{raw}"
        );
        assert!(
            raw.to_ascii_lowercase()
                .contains("authorization: bearer sk-secret-123")
        );
        let body = request_json(&raw);
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("from English into Korean")
        );
        assert_eq!(
            body["messages"][1],
            json!({"role": "user", "content": "Hello"})
        );
    }

    #[test]
    fn no_authorization_without_key() {
        let (base_url, server) = mock_server("200 OK", &ok_body("やあ"));
        let mut t = translator(base_url, None);
        assert_eq!(t.translate("Hi", None, "ja").unwrap(), "やあ");
        let raw = server.join().unwrap();
        assert!(!raw.to_ascii_lowercase().contains("authorization"), "{raw}");
        assert!(
            request_json(&raw)["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("into Japanese")
        );
    }

    #[test]
    fn second_request_carries_context() {
        let (base_url, server) = mock_server("200 OK", &ok_body("하나"));
        let mut t = translator(base_url, None);
        t.translate("one", Some("en"), "ko").unwrap();
        server.join().unwrap();
        // The context is sent to the next request.
        let body = t.request_body("two", Some("en"), "ko");
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("- one")
        );
    }

    #[test]
    fn http_error_has_status_and_server_text_but_no_key() {
        let body =
            json!({"error": {"message": "Incorrect API key provided: sk-secret-123."}}).to_string();
        let (base_url, server) = mock_server("401 Unauthorized", &body);
        let mut t = translator(base_url, Some("sk-secret-123"));
        let err = t
            .translate("Hello", Some("en"), "ko")
            .unwrap_err()
            .to_string();
        server.join().unwrap();
        assert!(err.contains("401"), "{err}");
        assert!(err.contains("Incorrect API key provided"), "{err}");
        assert!(!err.contains("sk-secret-123"), "{err}");
    }

    #[test]
    fn a_key_across_the_length_limit_is_not_kept() {
        let key = "sk-test-1234567890abcdefghij";
        let message = format!("{}{key} is wrong", "x".repeat(MAX_ERROR_CHARS - 20));
        let body = json!({"error": {"message": message}}).to_string();
        let (base_url, server) = mock_server("401 Unauthorized", &body);
        let mut t = translator(base_url, Some(key));
        let err = t
            .translate("Hello", Some("en"), "ko")
            .unwrap_err()
            .to_string();
        server.join().unwrap();
        assert!(!err.contains("sk-test"), "{err}");
        assert!(err.contains("***"), "{err}");
    }

    #[test]
    fn bad_responses_are_errors() {
        for body in ["not json", "{\"choices\": []}", &ok_body("  ")] {
            let (base_url, server) = mock_server("200 OK", body);
            let mut t = translator(base_url, None);
            assert!(t.translate("Hello", Some("en"), "ko").is_err(), "{body}");
            server.join().unwrap();
        }
    }

    #[test]
    fn connection_error_is_korean_and_has_no_key() {
        // Bind and drop to get a closed port.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut t = translator(
            format!("http://user:pw@127.0.0.1:{port}/v1"),
            Some("sk-secret-123"),
        );
        let err = t
            .translate("Hello", Some("en"), "ko")
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("번역 서버에 연결하지 못했습니다"), "{err}");
        assert!(
            !err.contains("sk-secret-123") && !err.contains("pw@"),
            "{err}"
        );
    }

    #[test]
    fn label_has_model() {
        assert_eq!(
            translator("http://x/v1".into(), None).label(),
            "사용자 지정: gpt-test"
        );
    }

    #[test]
    fn error_text_forms() {
        assert_eq!(error_text("{\"error\":\"boom\"}"), "boom");
        assert_eq!(error_text("{\"error\":{\"message\":\"bad\"}}"), "bad");
        assert_eq!(error_text("plain text"), "plain text");
    }
}
