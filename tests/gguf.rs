use std::io::Write;
use std::path::Path;

use pyxis::gguf::{GgmlType, GgufFile};

const GGUF_PATH: &str = "/Users/mizzy/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

fn load_test_gguf() -> Option<GgufFile> {
    let path = Path::new(GGUF_PATH);
    if !path.exists() {
        return None;
    }

    Some(GgufFile::parse(path).expect("parse gguf file"))
}

#[test]
fn parse_reads_header() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.tensors.len(), 310);
    assert_eq!(gguf.metadata.len(), 36);
}

#[test]
fn parse_reads_metadata() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    assert_eq!(gguf.get_str("general.architecture"), Some("qwen3"));
    assert_eq!(gguf.get_u32("qwen3.block_count"), Some(28));
    assert_eq!(gguf.get_u32("qwen3.embedding_length"), Some(1024));
}

#[test]
fn parse_reads_tensor_info() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    let first = gguf.tensors.first().expect("first tensor");
    assert_eq!(first.name, "output_norm.weight");
    assert_eq!(first.dims, [1024]);
    assert_eq!(first.tensor_type, GgmlType::F32);

    let token_embd = gguf
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight tensor");
    assert_eq!(token_embd.tensor_type, GgmlType::Q8_0);
}

#[test]
fn parse_errors_on_invalid_magic() {
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    file.write_all(b"NOPE").expect("write invalid magic");

    let result = GgufFile::parse(file.path());

    assert!(result.is_err());
}

#[test]
fn get_u32_returns_value() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    assert_eq!(gguf.get_u32("qwen3.block_count"), Some(28));
}

#[test]
fn tensor_info_finds_by_name() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    let tensor = gguf
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight tensor");

    assert_eq!(tensor.dims, [1024, 151669]);
}
