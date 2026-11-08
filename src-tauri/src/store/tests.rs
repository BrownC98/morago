use super::*;
use crate::types::{SourceKind, TranslationState};

const ID: &str = "20261108-140212-ab12";

fn meta(id: &str, started_at: &str) -> SessionMeta {
    SessionMeta {
        id: id.into(),
        title: "주간 회의".into(),
        started_at: started_at.into(),
        ended_at: None,
        target_language: "ko".into(),
        asr_label: "Local Whisper small".into(),
        translator_label: "Local Qwen2.5 3B".into(),
        speaker_names: BTreeMap::new(),
    }
}

fn utt(
    id: u64,
    speaker: &str,
    start_ms: u64,
    end_ms: u64,
    translation: TranslationState,
) -> Utterance {
    Utterance {
        id,
        source: SourceKind::System,
        speaker: speaker.into(),
        start_ms,
        end_ms,
        language: Some("en".into()),
        original: format!("line {id}"),
        translation,
    }
}

fn done(text: &str) -> TranslationState {
    TranslationState::Done {
        target: "ko".into(),
        text: text.into(),
    }
}

fn store() -> (tempfile::TempDir, SessionStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(&dir.path().join("sessions"));
    (dir, store)
}

#[test]
fn create_append_replace_load() {
    let (_d, store) = store();
    let m = meta(ID, "2026-11-08T14:02:12+09:00");
    let mut w = store.create(&m).unwrap();
    w.append(&utt(1, "S1", 0, 1000, TranslationState::Pending))
        .unwrap();
    w.append(&utt(2, "S2", 1000, 2500, TranslationState::Pending))
        .unwrap();
    w.append(&utt(1, "S1", 0, 1000, done("하나"))).unwrap();

    // Readable while the session runs.
    let (loaded, list) = store.load(ID).unwrap();
    assert_eq!(loaded, m);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].translation, done("하나"));
    assert_eq!(list[1].translation, TranslationState::Pending);

    w.finish("2026-11-08T15:00:00+09:00").unwrap();
    let (loaded, _) = store.load(ID).unwrap();
    assert_eq!(
        loaded.ended_at.as_deref(),
        Some("2026-11-08T15:00:00+09:00")
    );

    // The same id cannot be made twice.
    assert!(store.create(&m).is_err());
}

#[test]
fn torn_last_line_is_skipped() {
    let (_d, store) = store();
    let mut w = store
        .create(&meta(ID, "2026-11-08T14:02:12+09:00"))
        .unwrap();
    w.append(&utt(1, "S1", 0, 1000, done("하나"))).unwrap();
    w.append(&utt(2, "S1", 1000, 2000, done("둘"))).unwrap();
    drop(w);
    let path = store.root().join(ID).join(UTTERANCES);
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(br#"{"id":3,"source":"mic","speak"#).unwrap();
    drop(f);
    let (_, list) = store.load(ID).unwrap();
    assert_eq!(list.iter().map(|u| u.id).collect::<Vec<_>>(), [1, 2]);
}

#[test]
fn line_after_a_failed_write_starts_on_a_new_line() {
    let (_d, store) = store();
    let mut w = store
        .create(&meta(ID, "2026-11-08T14:02:12+09:00"))
        .unwrap();
    w.append(&utt(1, "S1", 0, 1000, done("하나"))).unwrap();
    // A write that failed half way (disk full).
    w.file.write_all(br#"{"id":2,"sou"#).unwrap();
    w.torn = true;
    w.append(&utt(3, "S1", 2000, 3000, done("셋"))).unwrap();
    let (_, list) = store.load(ID).unwrap();
    assert_eq!(list.iter().map(|u| u.id).collect::<Vec<_>>(), [1, 3]);
}

#[test]
fn list_is_newest_first_with_summary_and_skips_broken() {
    let (_d, store) = store();
    // Same local clock, other offsets: B (05:00 UTC) is newer than A (03:00 UTC).
    let a = "20261108-120000-aaaa";
    let b = "20261108-140000-bbbb";
    let c = "20261107-090000-cccc";
    let mut w = store.create(&meta(a, "2026-11-08T12:00:00+09:00")).unwrap();
    w.append(&utt(1, "S1", 0, 4000, done("x"))).unwrap();
    w.append(&utt(2, "S2", 4000, 9000, done("y"))).unwrap();
    w.append(&utt(3, "S1", 9000, 7000, done("z"))).unwrap();
    w.finish("2026-11-08T12:10:00+09:00").unwrap();
    store.create(&meta(b, "2026-11-08T14:00:00+09:00")).unwrap();
    store.create(&meta(c, "2026-11-07T09:00:00+09:00")).unwrap();
    // Broken folders: no meta.json, corrupt meta.json, a name that is no id.
    std::fs::create_dir(store.root().join("20261108-150000-dddd")).unwrap();
    let bad = store.root().join("20261108-160000-eeee");
    std::fs::create_dir(&bad).unwrap();
    std::fs::write(bad.join(META), "{").unwrap();
    std::fs::create_dir(store.root().join("notes")).unwrap();

    let list = store.list().unwrap();
    let ids: Vec<_> = list.iter().map(|s| s.meta.id.as_str()).collect();
    assert_eq!(ids, [b, a, c]);
    let sa = &list[1];
    assert_eq!(sa.duration_ms, 9000);
    assert_eq!(sa.utterance_count, 3);
    assert_eq!(sa.speaker_count, 2);
    assert_eq!(list[0].utterance_count, 0);
    assert_eq!(list[0].duration_ms, 0);
}

#[test]
fn list_of_missing_root_is_empty() {
    let (_d, store) = store();
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn update_meta_changes_only_title_and_names() {
    let (_d, store) = store();
    let m = meta(ID, "2026-11-08T14:02:12+09:00");
    let w = store.create(&m).unwrap();
    let mut changed = m.clone();
    changed.title = "새 제목".into();
    changed.speaker_names.insert("S2".into(), "나".into());
    changed.asr_label = "hacked".into();
    changed.started_at = "2000-01-01T00:00:00Z".into();
    store.update_meta(&changed).unwrap();
    // finish keeps the title set while recording.
    w.finish("2026-11-08T15:00:00+09:00").unwrap();
    let (loaded, _) = store.load(ID).unwrap();
    assert_eq!(loaded.title, "새 제목");
    assert_eq!(
        loaded.speaker_names.get("S2").map(String::as_str),
        Some("나")
    );
    assert_eq!(loaded.asr_label, m.asr_label);
    assert_eq!(loaded.started_at, m.started_at);
    assert!(loaded.ended_at.is_some());
}

#[test]
fn delete_removes_the_folder() {
    let (_d, store) = store();
    store
        .create(&meta(ID, "2026-11-08T14:02:12+09:00"))
        .unwrap();
    store.delete(ID).unwrap();
    assert!(!store.root().join(ID).exists());
    store.delete(ID).unwrap();
}

#[test]
fn ids_are_validated_against_traversal() {
    let (d, store) = store();
    std::fs::create_dir_all(store.root()).unwrap();
    let outside = d.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join(META), "{}").unwrap();
    for id in [
        "",
        "..",
        "../outside",
        "..\\outside",
        "20261108-140212-ab12/../../outside",
        "20261108-140212-ab1/",
        "/etc",
        "C:\\Windows",
        "20261108-140212-AB12",
        "20261108-140212-ab12 ",
        "2026110x-140212-ab12",
        "20261108_140212-ab12",
        "20261108-140212-ab123",
    ] {
        assert!(!valid_id(id), "{id:?}");
        assert!(store.load(id).is_err(), "{id:?}");
        assert!(store.export(id, ExportFormat::Json).is_err(), "{id:?}");
        assert!(store.delete(id).is_err(), "{id:?}");
        assert!(store.create(&meta(id, "x")).is_err(), "{id:?}");
        assert!(store.update_meta(&meta(id, "x")).is_err(), "{id:?}");
    }
    assert!(outside.join(META).exists());
    assert!(valid_id(ID));
    assert!(valid_id(&new_session_id()));
}

fn export_fixture(store: &SessionStore) {
    let mut m = meta(ID, "2026-11-08T14:02:12+09:00");
    m.speaker_names.insert("S2".into(), "나".into());
    let mut w = store.create(&m).unwrap();
    let mut first = utt(1, "S1", 83_000, 91_250, done("모두 안녕하세요."));
    first.original = "Hello\n everyone.".into();
    w.append(&first).unwrap();
    let mut second = utt(
        2,
        "S2",
        3_723_004,
        3_725_000,
        TranslationState::SameLanguage,
    );
    second.language = Some("ko".into());
    second.original = "네, 시작하죠.".into();
    w.append(&second).unwrap();
    let mut third = utt(
        3,
        "S10",
        3_726_000,
        3_727_500,
        TranslationState::Failed {
            message: "시간 초과".into(),
        },
    );
    third.original = "One more.".into();
    w.append(&third).unwrap();
    w.finish("2026-11-08T15:04:00+09:00").unwrap();
}

#[test]
fn export_markdown_golden() {
    let (_d, store) = store();
    export_fixture(&store);
    let text = store.export(ID, ExportFormat::Markdown).unwrap();
    // Header lines end with two spaces (Markdown line break).
    let expected = concat!(
        "# 주간 회의\n",
        "\n",
        "- 날짜: 2026-11-08 14:02\n",
        "- 원문 언어: en, ko\n",
        "- 번역 언어: ko\n",
        "- 음성 인식: Local Whisper small\n",
        "- 번역: Local Qwen2.5 3B\n",
        "\n",
        "**화자 1** · 00:01:23–00:01:31  \n",
        "Hello everyone.\n",
        "> 모두 안녕하세요.\n",
        "\n",
        "**나** · 01:02:03–01:02:05  \n",
        "네, 시작하죠.\n",
        "\n",
        "**화자 10** · 01:02:06–01:02:07  \n",
        "One more.\n",
    );
    assert_eq!(text, expected);
}

#[test]
fn export_srt_golden() {
    let (_d, store) = store();
    export_fixture(&store);
    let text = store.export(ID, ExportFormat::Srt).unwrap();
    let expected = "1
00:01:23,000 --> 00:01:31,250
화자 1: Hello everyone.
모두 안녕하세요.

2
01:02:03,004 --> 01:02:05,000
나: 네, 시작하죠.

3
01:02:06,000 --> 01:02:07,500
화자 10: One more.

";
    assert_eq!(text, expected);
}

#[test]
fn export_json_golden() {
    let (_d, store) = store();
    export_fixture(&store);
    let text = store.export(ID, ExportFormat::Json).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["meta"]["id"], ID);
    assert_eq!(value["meta"]["speaker_names"]["S2"], "나");
    assert_eq!(value["utterances"].as_array().unwrap().len(), 3);
    assert_eq!(value["utterances"][0]["translation"]["state"], "done");
    assert_eq!(
        value["utterances"][0]["translation"]["text"],
        "모두 안녕하세요."
    );
    assert!(
        text.starts_with("{\n  \"meta\": {\n    \"id\": \"20261108-140212-ab12\",\n"),
        "{text}"
    );
}

#[test]
fn write_atomic_replaces_and_leaves_no_temp() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.json");
    write_atomic(&path, b"one").unwrap();
    write_atomic(&path, b"two").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"two");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}
