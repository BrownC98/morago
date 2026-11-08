//! OpenAI-compatible chat completions provider.

use super::{History, Translator, clean_output, system_prompt};
use crate::secrets::Secret;
use crate::settings::ReasoningEffort;
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
    /// Sent as `reasoning_effort` unless the server refused it.
    effort: Option<ReasoningEffort>,
    /// Optional parameters that the server refused. Left out of each request
    /// for the life of this translator.
    refused: Vec<Param>,
    /// The "not supported" message that waits for `take_notice`. Set once.
    notice: Option<String>,
}

/// An optional request parameter that a server can refuse with HTTP 400.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    /// `temperature`: OpenAI reasoning models accept only the default (1).
    Temperature,
    ReasoningEffort,
}

impl Param {
    /// The word that a server error names for this parameter.
    fn word(self) -> &'static str {
        match self {
            Param::Temperature => "temperature",
            Param::ReasoningEffort => "reasoning",
        }
    }
}

/// Shown once when a server refuses `reasoning_effort`.
const EFFORT_UNSUPPORTED: &str = "이 서버는 추론 강도 설정을 지원하지 않습니다";

impl CustomTranslator {
    pub(crate) fn new(
        endpoint: Endpoint,
        key: Option<Secret>,
        effort: Option<ReasoningEffort>,
    ) -> anyhow::Result<Self> {
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
            effort,
            refused: Vec::new(),
            notice: None,
        })
    }

    fn request_body(&self, text: &str, source: Option<&str>, target: &str) -> Value {
        let mut body = json!({
            "model": self.endpoint.model,
            "stream": false,
            "messages": [
                {"role": "system", "content": system_prompt(source, target, &self.history)},
                {"role": "user", "content": text},
            ],
        });
        for param in self.sent_params() {
            match param {
                Param::Temperature => body["temperature"] = json!(0.2),
                Param::ReasoningEffort => {
                    if let Some(effort) = self.effort {
                        body["reasoning_effort"] = json!(effort.as_str());
                    }
                }
            }
        }
        body
    }

    /// The optional parameters that the next request sends.
    fn sent_params(&self) -> Vec<Param> {
        let mut params = vec![Param::Temperature];
        if self.effort.is_some() {
            params.push(Param::ReasoningEffort);
        }
        params.retain(|p| !self.refused.contains(p));
        params
    }

    /// Sends one request. Returns the HTTP status and the body text.
    fn post(&self, body: &Value) -> anyhow::Result<(u16, String)> {
        let mut request = self.client.post(&self.url).json(body);
        if let Some(key) = &self.key {
            request = request.bearer_auth(key.expose());
        }
        let response = request.send().map_err(|e| {
            anyhow!(
                "번역 서버에 연결하지 못했습니다: {}",
                self.redact(&e.without_url().to_string())
            )
        })?;
        let status = response.status().as_u16();
        let text = response.text().map_err(|e| {
            anyhow!(
                "번역 서버의 응답을 읽지 못했습니다: {}",
                self.redact(&e.without_url().to_string())
            )
        })?;
        Ok((status, text))
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

/// The first parameter of `sent` that a 400 body names, in `error.param`
/// or in the error text.
fn refused_param(body: &str, sent: &[Param]) -> Option<Param> {
    let param = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/param")?
                .as_str()
                .map(str::to_ascii_lowercase)
        })
        .unwrap_or_default();
    let text = error_text(body).to_ascii_lowercase();
    sent.iter()
        .copied()
        .find(|p| param.contains(p.word()) || text.contains(p.word()))
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
        self.history.set_target(target_language);
        // A 400 that names an optional parameter (temperature, reasoning_effort):
        // the server does not accept it. Retry once without it, at most once per
        // parameter. Any other 400 (bad model, bad request) is reported as it is.
        let mut dropped = Vec::new();
        let (status, response_text) = loop {
            let body = self.request_body(text, source_language, target_language);
            let reply = match self.post(&body) {
                Ok(reply) => reply,
                Err(e) => {
                    self.refused.retain(|p| !dropped.contains(p));
                    return Err(e);
                }
            };
            let refused = (reply.0 == 400)
                .then(|| refused_param(&reply.1, &self.sent_params()))
                .flatten();
            match refused {
                Some(param) => {
                    dropped.push(param);
                    self.refused.push(param);
                }
                None => break reply,
            }
        };
        if (200..300).contains(&status) {
            // The parameters were the cause: they stay out for the rest of the session.
            if dropped.contains(&Param::ReasoningEffort) {
                self.notice = Some(EFFORT_UNSUPPORTED.to_string());
            }
        } else {
            self.refused.retain(|p| !dropped.contains(p));
        }
        let success = (200..300).contains(&status);
        if !success {
            return Err(anyhow!(
                "번역 서버 오류 (HTTP {}): {}",
                status,
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

    fn take_notice(&mut self) -> Option<String> {
        self.notice.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    /// Reads one HTTP request (head and body) and returns its raw text.
    fn read_request(stream: &mut std::net::TcpStream) -> String {
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
        String::from_utf8_lossy(&data).to_string()
    }

    /// HTTP server: answers one request per entry of `replies` (status, body),
    /// then returns all raw requests.
    fn mock_server_seq(replies: Vec<(&str, String)>) -> (String, JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let replies: Vec<String> = replies
            .into_iter()
            .map(|(status, body)| {
                format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            })
            .collect();
        let handle = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut stream, _) = listener.accept().unwrap();
                requests.push(read_request(&mut stream));
                stream.write_all(reply.as_bytes()).unwrap();
            }
            requests
        });
        (base_url, handle)
    }

    /// One-shot HTTP server: answers one request, returns the raw request.
    fn mock_server(status: &str, body: &str) -> (String, JoinHandle<String>) {
        let (base_url, handle) = mock_server_seq(vec![(status, body.to_string())]);
        let handle = std::thread::spawn(move || handle.join().unwrap().remove(0));
        (base_url, handle)
    }

    fn translator(base_url: String, key: Option<&str>) -> CustomTranslator {
        translator_with(base_url, key, None)
    }

    fn translator_with(
        base_url: String,
        key: Option<&str>,
        effort: Option<ReasoningEffort>,
    ) -> CustomTranslator {
        let endpoint = Endpoint {
            base_url,
            model: "gpt-test".into(),
        };
        CustomTranslator::new(endpoint, key.map(|k| Secret::new(k.into())), effort).unwrap()
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
    fn context_resets_on_target_change_and_new_session() {
        let (base_url, server) = mock_server_seq(vec![
            ("200 OK", ok_body("하나")),
            ("200 OK", ok_body("ふた")),
            ("200 OK", ok_body("셋")),
        ]);
        let mut t = translator(base_url.clone(), None);
        t.translate("one", Some("en"), "ko").unwrap();
        t.translate("two", Some("en"), "ja").unwrap();
        // A new session builds a new translator.
        let mut next = translator(base_url, None);
        next.translate("three", Some("en"), "ko").unwrap();
        let requests = server.join().unwrap();
        let system = |i: usize| request_json(&requests[i])["messages"][0]["content"].clone();
        assert!(!system(1).as_str().unwrap().contains("Earlier lines"));
        assert!(!system(2).as_str().unwrap().contains("Earlier lines"));
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

    #[test]
    fn body_has_no_reasoning_effort_by_default() {
        let (base_url, server) = mock_server("200 OK", &ok_body("하나"));
        let mut t = translator(base_url, None);
        t.translate("one", Some("en"), "ko").unwrap();
        let body = request_json(&server.join().unwrap());
        assert!(body.get("reasoning_effort").is_none(), "{body}");
    }

    #[test]
    fn body_has_the_chosen_reasoning_effort() {
        for effort in [
            ReasoningEffort::Off,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ] {
            let (base_url, server) = mock_server("200 OK", &ok_body("하나"));
            let mut t = translator_with(base_url, None, Some(effort));
            t.translate("one", Some("en"), "ko").unwrap();
            let body = request_json(&server.join().unwrap());
            assert_eq!(body["reasoning_effort"], effort.as_str(), "{body}");
            assert_eq!(t.take_notice(), None);
        }
    }

    #[test]
    fn a_400_about_reasoning_retries_once_without_it_and_notes_once() {
        let refusal = json!({"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}})
            .to_string();
        let (base_url, server) = mock_server_seq(vec![
            ("400 Bad Request", refusal),
            ("200 OK", ok_body("하나")),
            ("200 OK", ok_body("둘")),
        ]);
        let mut t = translator_with(base_url, None, Some(ReasoningEffort::High));
        assert_eq!(t.translate("one", Some("en"), "ko").unwrap(), "하나");
        assert_eq!(t.take_notice().as_deref(), Some(EFFORT_UNSUPPORTED));
        // The next request does not send the field, and there is no second notice.
        assert_eq!(t.translate("two", Some("en"), "ko").unwrap(), "둘");
        assert_eq!(t.take_notice(), None);
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(request_json(&requests[0])["reasoning_effort"], "high");
        assert!(request_json(&requests[1]).get("reasoning_effort").is_none());
        assert!(request_json(&requests[2]).get("reasoning_effort").is_none());
    }

    #[test]
    fn a_400_about_something_else_is_an_error_and_keeps_the_field() {
        let refusal = json!({"error": {"message": "model not found"}}).to_string();
        let (base_url, server) = mock_server_seq(vec![
            ("400 Bad Request", refusal),
            ("200 OK", ok_body("하나")),
        ]);
        let mut t = translator_with(base_url, None, Some(ReasoningEffort::Low));
        let err = t
            .translate("one", Some("en"), "ko")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("400") && err.contains("model not found"),
            "{err}"
        );
        assert_eq!(t.take_notice(), None);
        // The setting stays: the next request still sends it.
        t.translate("one", Some("en"), "ko").unwrap();
        let requests = server.join().unwrap();
        assert_eq!(request_json(&requests[1])["reasoning_effort"], "low");
    }

    #[test]
    fn a_failed_retry_reports_its_error_and_keeps_the_field() {
        let refusal = json!({"error": {"message": "unknown field reasoning_effort"}}).to_string();
        let (base_url, server) = mock_server_seq(vec![
            ("400 Bad Request", refusal),
            ("500 Internal Server Error", "{\"error\":\"boom\"}".into()),
        ]);
        let mut t = translator_with(base_url, None, Some(ReasoningEffort::Medium));
        let err = t
            .translate("one", Some("en"), "ko")
            .unwrap_err()
            .to_string();
        server.join().unwrap();
        assert!(err.contains("500") && err.contains("boom"), "{err}");
        assert_eq!(t.take_notice(), None);
        assert_eq!(t.effort, Some(ReasoningEffort::Medium));
        assert!(t.refused.is_empty());
    }

    #[test]
    fn a_400_about_temperature_retries_once_without_it_and_remembers() {
        let refusal = json!({"error": {
            "message": "Unsupported value: 'temperature' does not support 0.2 with this model. Only the default (1) value is supported.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_value"
        }})
        .to_string();
        let (base_url, server) = mock_server_seq(vec![
            ("400 Bad Request", refusal),
            ("200 OK", ok_body("하나")),
            ("200 OK", ok_body("둘")),
        ]);
        let mut t = translator_with(base_url, None, Some(ReasoningEffort::Low));
        assert_eq!(t.translate("one", Some("en"), "ko").unwrap(), "하나");
        // Temperature is not a user setting: no notice.
        assert_eq!(t.take_notice(), None);
        assert_eq!(t.translate("two", Some("en"), "ko").unwrap(), "둘");
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(request_json(&requests[0])["temperature"], 0.2);
        for raw in &requests[1..] {
            let body = request_json(raw);
            assert!(body.get("temperature").is_none(), "{body}");
            assert_eq!(body["reasoning_effort"], "low", "{body}");
        }
    }

    #[test]
    fn a_server_that_refuses_both_parameters_gets_neither() {
        let temperature = json!({"error": {"message": "Unsupported parameter: 'temperature'", "code": "unsupported_parameter"}})
            .to_string();
        let reasoning = json!({"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}})
            .to_string();
        let (base_url, server) = mock_server_seq(vec![
            ("400 Bad Request", temperature),
            ("400 Bad Request", reasoning),
            ("200 OK", ok_body("하나")),
            ("200 OK", ok_body("둘")),
        ]);
        let mut t = translator_with(base_url, None, Some(ReasoningEffort::High));
        assert_eq!(t.translate("one", Some("en"), "ko").unwrap(), "하나");
        assert_eq!(t.take_notice().as_deref(), Some(EFFORT_UNSUPPORTED));
        assert_eq!(t.translate("two", Some("en"), "ko").unwrap(), "둘");
        assert_eq!(t.take_notice(), None);
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 4);
        let first = request_json(&requests[0]);
        assert!(first.get("temperature").is_some() && first.get("reasoning_effort").is_some());
        let second = request_json(&requests[1]);
        assert!(second.get("temperature").is_none() && second.get("reasoning_effort").is_some());
        for raw in &requests[2..] {
            let body = request_json(raw);
            assert!(
                body.get("temperature").is_none() && body.get("reasoning_effort").is_none(),
                "{body}"
            );
        }
    }

    #[test]
    fn a_second_400_about_the_same_parameter_is_reported() {
        let refusal = json!({"error": {"message": "temperature is not supported"}}).to_string();
        let (base_url, server) = mock_server_seq(vec![
            ("400 Bad Request", refusal.clone()),
            ("400 Bad Request", refusal),
        ]);
        let mut t = translator(base_url, None);
        let err = t
            .translate("one", Some("en"), "ko")
            .unwrap_err()
            .to_string();
        assert!(err.contains("400"), "{err}");
        assert_eq!(server.join().unwrap().len(), 2);
        assert!(t.refused.is_empty());
    }
}
