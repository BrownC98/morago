//! Custom ASR: an OpenAI-compatible `POST {base_url}/audio/transcriptions` endpoint.

use super::{Asr, wav::encode_wav};
use crate::secrets::Secret;
use crate::types::{Endpoint, SpeechSegment, Transcript};
use anyhow::{Context, bail};
use std::time::Duration;

pub struct CustomAsr {
    client: reqwest::blocking::Client,
    url: String,
    model: String,
    key: Option<Secret>,
    /// Meeting language (ISO 639-1): sent as `language`; `None` = server detects.
    language: Option<String>,
}

impl CustomAsr {
    pub fn new(
        endpoint: &Endpoint,
        key: Option<Secret>,
        language: Option<&str>,
    ) -> anyhow::Result<Self> {
        let base = endpoint.base_url.trim().trim_end_matches('/');
        if base.is_empty() || !(base.starts_with("http://") || base.starts_with("https://")) {
            bail!(
                "ASR 서버 주소가 올바르지 않습니다. http:// 또는 https:// 로 시작하는 주소를 입력하세요."
            );
        }
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(60))
            .build()
            .context("HTTP 클라이언트를 만들 수 없습니다")?;
        Ok(Self {
            client,
            url: format!("{base}/audio/transcriptions"),
            model: endpoint.model.clone(),
            key,
            language: language.map(str::to_string),
        })
    }

    fn request(
        &self,
        wav: &[u8],
        response_format: &str,
    ) -> anyhow::Result<reqwest::blocking::Response> {
        let part = reqwest::blocking::multipart::Part::bytes(wav.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")?;
        let mut form = reqwest::blocking::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone())
            .text("response_format", response_format.to_string());
        if let Some(language) = &self.language {
            form = form.text("language", language.clone());
        }
        let mut req = self.client.post(&self.url).multipart(form);
        if let Some(key) = &self.key {
            req = req.bearer_auth(key.expose());
        }
        req.send().map_err(|e| {
            // `without_url` keeps credentials in a URL out of the message.
            anyhow::anyhow!(
                "ASR 서버에 연결하지 못했습니다: {}. 주소와 네트워크를 확인하세요.",
                e.without_url()
            )
        })
    }
}

impl Asr for CustomAsr {
    fn transcribe(&mut self, segment: &SpeechSegment) -> anyhow::Result<Transcript> {
        let wav = encode_wav(&segment.samples);
        let mut resp = self.request(&wav, "verbose_json")?;
        let mut verbose = true;
        // Some servers accept only `json` or `text`: retry once without language.
        if matches!(resp.status().as_u16(), 400 | 415 | 422) {
            resp = self.request(&wav, "json")?;
            verbose = false;
        }
        let status = resp.status();
        if !status.is_success() {
            bail!("{}", http_error_message(status.as_u16()));
        }
        let body = resp.text().context("ASR 서버 응답을 읽지 못했습니다")?;
        let mut t = parse_response(&body)?;
        if self.language.is_some() {
            t.language = self.language.clone();
        } else if !verbose {
            t.language = None;
        }
        Ok(t)
    }

    fn label(&self) -> String {
        format!("사용자 지정: {}", self.model)
    }
}

/// Korean message for a failed request. The response body is not used: a
/// server can echo request details.
fn http_error_message(status: u16) -> String {
    let hint = match status {
        401 | 403 => "API 키를 확인하세요.",
        404 => "서버 주소와 모델 이름을 확인하세요.",
        413 => "오디오가 너무 큽니다.",
        429 => "요청 한도를 넘었습니다. 잠시 뒤 다시 시도하세요.",
        500..=599 => "서버 오류입니다. 잠시 뒤 다시 시도하세요.",
        _ => "설정을 확인하세요.",
    };
    format!("ASR 서버 오류 (HTTP {status}): {hint}")
}

/// Parses `verbose_json` or `json`: `{"text": "...", "language": "english"}`.
pub fn parse_response(body: &str) -> anyhow::Result<Transcript> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|_| {
        anyhow::anyhow!(
            "ASR 서버가 JSON이 아닌 응답을 보냈습니다. 서버가 OpenAI 호환인지 확인하세요."
        )
    })?;
    let Some(text) = v.get("text").and_then(|t| t.as_str()) else {
        bail!("ASR 서버 응답에 text 항목이 없습니다. 서버가 OpenAI 호환인지 확인하세요.");
    };
    let language = v
        .get("language")
        .and_then(|l| l.as_str())
        .and_then(language_code);
    Ok(Transcript {
        text: text.trim().to_string(),
        language,
    })
}

/// Maps a language name ("english", "Korean") or code ("en", "ko-KR") to ISO 639-1.
pub fn language_code(name: &str) -> Option<String> {
    let lower = name.trim().to_lowercase();
    if lower.is_empty() {
        return None;
    }
    if let Some((_, code)) = LANGUAGES.iter().find(|(n, _)| *n == lower) {
        return Some((*code).to_string());
    }
    let primary = lower.split(['-', '_']).next().unwrap_or("");
    if primary.len() == 2 && LANGUAGES.iter().any(|(_, c)| *c == primary) {
        return Some(primary.to_string());
    }
    None
}

/// Whisper language names and ISO 639-1 codes (OpenAI-compatible services).
const LANGUAGES: &[(&str, &str)] = &[
    ("english", "en"),
    ("chinese", "zh"),
    ("german", "de"),
    ("spanish", "es"),
    ("russian", "ru"),
    ("korean", "ko"),
    ("french", "fr"),
    ("japanese", "ja"),
    ("portuguese", "pt"),
    ("turkish", "tr"),
    ("polish", "pl"),
    ("catalan", "ca"),
    ("dutch", "nl"),
    ("arabic", "ar"),
    ("swedish", "sv"),
    ("italian", "it"),
    ("indonesian", "id"),
    ("hindi", "hi"),
    ("finnish", "fi"),
    ("vietnamese", "vi"),
    ("hebrew", "he"),
    ("ukrainian", "uk"),
    ("greek", "el"),
    ("malay", "ms"),
    ("czech", "cs"),
    ("romanian", "ro"),
    ("danish", "da"),
    ("hungarian", "hu"),
    ("tamil", "ta"),
    ("norwegian", "no"),
    ("thai", "th"),
    ("urdu", "ur"),
    ("croatian", "hr"),
    ("bulgarian", "bg"),
    ("lithuanian", "lt"),
    ("latin", "la"),
    ("maori", "mi"),
    ("malayalam", "ml"),
    ("welsh", "cy"),
    ("slovak", "sk"),
    ("telugu", "te"),
    ("persian", "fa"),
    ("latvian", "lv"),
    ("bengali", "bn"),
    ("serbian", "sr"),
    ("azerbaijani", "az"),
    ("slovenian", "sl"),
    ("kannada", "kn"),
    ("estonian", "et"),
    ("macedonian", "mk"),
    ("breton", "br"),
    ("basque", "eu"),
    ("icelandic", "is"),
    ("armenian", "hy"),
    ("nepali", "ne"),
    ("mongolian", "mn"),
    ("bosnian", "bs"),
    ("kazakh", "kk"),
    ("albanian", "sq"),
    ("swahili", "sw"),
    ("galician", "gl"),
    ("marathi", "mr"),
    ("punjabi", "pa"),
    ("sinhala", "si"),
    ("khmer", "km"),
    ("shona", "sn"),
    ("yoruba", "yo"),
    ("somali", "so"),
    ("afrikaans", "af"),
    ("occitan", "oc"),
    ("georgian", "ka"),
    ("belarusian", "be"),
    ("tajik", "tg"),
    ("sindhi", "sd"),
    ("gujarati", "gu"),
    ("amharic", "am"),
    ("yiddish", "yi"),
    ("lao", "lo"),
    ("uzbek", "uz"),
    ("faroese", "fo"),
    ("haitian creole", "ht"),
    ("pashto", "ps"),
    ("turkmen", "tk"),
    ("nynorsk", "nn"),
    ("maltese", "mt"),
    ("sanskrit", "sa"),
    ("luxembourgish", "lb"),
    ("myanmar", "my"),
    ("tibetan", "bo"),
    ("tagalog", "tl"),
    ("malagasy", "mg"),
    ("assamese", "as"),
    ("tatar", "tt"),
    ("hawaiian", "haw"),
    ("lingala", "ln"),
    ("hausa", "ha"),
    ("bashkir", "ba"),
    ("javanese", "jw"),
    ("sundanese", "su"),
    ("cantonese", "yue"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    #[test]
    fn language_names_map_to_iso() {
        assert_eq!(language_code("english").as_deref(), Some("en"));
        assert_eq!(language_code("Korean").as_deref(), Some("ko"));
        assert_eq!(language_code(" japanese ").as_deref(), Some("ja"));
        assert_eq!(language_code("zh").as_deref(), Some("zh"));
        assert_eq!(language_code("en-US").as_deref(), Some("en"));
        assert_eq!(language_code("klingon"), None);
        assert_eq!(language_code(""), None);
    }

    #[test]
    fn parses_verbose_and_plain_json() {
        let t = parse_response(r#"{"task":"transcribe","language":"korean","duration":2.1,"text":" 안녕하세요 ","segments":[]}"#)
            .unwrap();
        assert_eq!(t.text, "안녕하세요");
        assert_eq!(t.language.as_deref(), Some("ko"));
        let t = parse_response(r#"{"text":"hello"}"#).unwrap();
        assert_eq!(t.language, None);
        assert!(parse_response("not json").is_err());
        assert!(parse_response(r#"{"error":"x"}"#).is_err());
    }

    struct Captured {
        head: String,
        body: Vec<u8>,
    }

    /// One-shot mock server: answers each request with the next canned reply
    /// `(status, body)` and sends every received request to the channel.
    fn mock(replies: Vec<(u16, &'static str)>) -> (String, mpsc::Receiver<Captured>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for (status, reply) in replies {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut head = String::new();
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                    head.push_str(&line);
                }
                let mut body = vec![0u8; len];
                reader.read_exact(&mut body).unwrap();
                tx.send(Captured { head, body }).unwrap();
                let resp = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        (format!("http://127.0.0.1:{port}/v1"), rx)
    }

    fn segment() -> SpeechSegment {
        SpeechSegment {
            source: crate::types::SourceKind::Mic,
            start_ms: 0,
            end_ms: 100,
            samples: vec![0.1; 1600],
        }
    }

    fn endpoint(base_url: String) -> Endpoint {
        Endpoint {
            base_url,
            model: "whisper-1".into(),
        }
    }

    fn contains(hay: &[u8], needle: &str) -> bool {
        hay.windows(needle.len()).any(|w| w == needle.as_bytes())
    }

    #[test]
    fn sends_multipart_with_key_and_parses_language() {
        let (url, rx) = mock(vec![(
            200,
            r#"{"text":"hello there","language":"english"}"#,
        )]);
        let mut asr = CustomAsr::new(
            &endpoint(url + "/"),
            Some(Secret::new("sk-test-123".into())),
            None,
        )
        .unwrap();
        let t = asr.transcribe(&segment()).unwrap();
        assert_eq!(t.text, "hello there");
        assert_eq!(t.language.as_deref(), Some("en"));
        assert_eq!(asr.label(), "사용자 지정: whisper-1");
        let req = rx.recv().unwrap();
        let head = req.head.to_lowercase();
        assert!(head.starts_with("post /v1/audio/transcriptions "), "{head}");
        assert!(head.contains("authorization: bearer sk-test-123"));
        assert!(head.contains("multipart/form-data"));
        assert!(contains(&req.body, "name=\"file\""));
        assert!(contains(&req.body, "RIFF"));
        assert!(contains(&req.body, "name=\"model\""));
        assert!(contains(&req.body, "whisper-1"));
        assert!(contains(&req.body, "verbose_json"));
        assert!(!contains(&req.body, "name=\"language\""));
    }

    #[test]
    fn a_meeting_language_is_sent_and_returned() {
        let (url, rx) = mock(vec![(200, r#"{"text":"はい","language":"korean"}"#)]);
        let mut asr = CustomAsr::new(&endpoint(url), None, Some("ja")).unwrap();
        let t = asr.transcribe(&segment()).unwrap();
        assert_eq!(t.language.as_deref(), Some("ja"));
        let body = rx.recv().unwrap().body;
        assert!(contains(&body, "name=\"language\"\r\n\r\nja\r\n"));
    }

    #[test]
    fn no_key_means_no_authorization_header() {
        let (url, rx) = mock(vec![(200, r#"{"text":"x"}"#)]);
        let mut asr = CustomAsr::new(&endpoint(url), None, None).unwrap();
        asr.transcribe(&segment()).unwrap();
        let head = rx.recv().unwrap().head.to_lowercase();
        assert!(!head.contains("authorization"), "{head}");
    }

    #[test]
    fn retries_once_with_json_when_verbose_json_is_rejected() {
        let (url, rx) = mock(vec![
            (400, r#"{"error":"unsupported response_format"}"#),
            (200, r#"{"text":"hello","language":"english"}"#),
        ]);
        let mut asr = CustomAsr::new(&endpoint(url), None, None).unwrap();
        let t = asr.transcribe(&segment()).unwrap();
        assert_eq!(t.text, "hello");
        assert_eq!(t.language, None);
        assert!(contains(&rx.recv().unwrap().body, "verbose_json"));
        let second = rx.recv().unwrap().body;
        assert!(!contains(&second, "verbose_json"));
        assert!(contains(&second, "\r\n\r\njson\r\n"));
    }

    #[test]
    fn error_names_status_and_never_the_key() {
        let (url, _rx) = mock(vec![(
            401,
            r#"{"error":{"message":"bad key sk-secret-999"}}"#,
        )]);
        let mut asr = CustomAsr::new(
            &endpoint(url),
            Some(Secret::new("sk-secret-999".into())),
            None,
        )
        .unwrap();
        let err = asr.transcribe(&segment()).unwrap_err().to_string();
        assert!(err.contains("401"), "{err}");
        assert!(err.contains("API 키"), "{err}");
        assert!(!err.contains("sk-secret-999"), "{err}");
    }

    #[test]
    fn rejects_bad_base_url() {
        assert!(CustomAsr::new(&endpoint("localhost:8000".into()), None, None).is_err());
        assert!(CustomAsr::new(&endpoint("".into()), None, None).is_err());
    }
}
