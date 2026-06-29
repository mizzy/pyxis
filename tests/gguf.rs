use std::io::Write;
use std::path::Path;

use pyxis::gguf::{GgmlType, GgufFile};
use pyxis::safetensors::TensorStore;
use pyxis::weights::Weights;

const GGUF_PATH: &str = "/Users/mizzy/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

fn load_test_gguf() -> Option<GgufFile> {
    let path = Path::new(GGUF_PATH);
    if !path.exists() {
        return None;
    }

    Some(GgufFile::parse(path).expect("parse gguf file"))
}

fn write_string(file: &mut impl Write, value: &str) {
    file.write_all(&(value.len() as u64).to_le_bytes())
        .expect("write string len");
    file.write_all(value.as_bytes()).expect("write string");
}

fn write_minimal_gguf(
    file: &mut impl Write,
    name: &str,
    dims: &[u64],
    tensor_type: u32,
    data: &[u8],
) {
    file.write_all(b"GGUF").expect("write magic");
    file.write_all(&3u32.to_le_bytes()).expect("write version");
    file.write_all(&1u64.to_le_bytes())
        .expect("write tensor count");
    file.write_all(&0u64.to_le_bytes())
        .expect("write metadata count");

    write_string(file, name);
    file.write_all(&(dims.len() as u32).to_le_bytes())
        .expect("write dim count");
    for dim in dims {
        file.write_all(&dim.to_le_bytes()).expect("write dim");
    }
    file.write_all(&tensor_type.to_le_bytes())
        .expect("write tensor type");
    file.write_all(&0u64.to_le_bytes()).expect("write offset");

    let header_len = 4 + 4 + 8 + 8 + 8 + name.len() + 4 + dims.len() * 8 + 4 + 8;
    let data_offset = (header_len + 31) & !31;
    file.write_all(&vec![0; data_offset - header_len])
        .expect("write padding");
    file.write_all(data).expect("write tensor data");
}

#[test]
fn q8_0_dequantize_known_values() {
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    for value in -16i8..16 {
        data.push(value as u8);
    }
    write_minimal_gguf(&mut file, "q8.weight", &[32], 8, &data);
    file.flush().expect("flush gguf");

    let gguf = GgufFile::parse(file.path()).expect("parse gguf");
    let values = gguf.tensor_f32("q8.weight").expect("read q8 tensor");

    let expected: Vec<f32> = (-16i8..16).map(|value| value as f32 * 0.5).collect();
    assert_eq!(values, expected);
}

#[test]
fn q8_0_tensor_weights_returns_f32() {
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
    for value in 0i8..32 {
        data.push(value as u8);
    }
    write_minimal_gguf(&mut file, "q8.weight", &[32], 8, &data);
    file.flush().expect("flush gguf");

    let gguf = GgufFile::parse(file.path()).expect("parse gguf");
    let weights = gguf.tensor_weights("q8.weight").expect("read q8 weights");

    let Weights::F32(values) = weights else {
        panic!("expected f32 weights");
    };
    assert_eq!(values[0], 0.0);
    assert_eq!(values[31], 31.0 * 0.25);
}

#[test]
fn tensor_store_loads_gguf_file() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("tiny.gguf");
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let data: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut file = std::fs::File::create(&path).expect("create gguf");
    write_minimal_gguf(&mut file, "output_norm.weight", &[4], 0, &data);
    file.flush().expect("flush gguf");

    let store = TensorStore::load(&path).expect("load gguf store");

    assert!(matches!(store, TensorStore::Gguf(_)));
    assert!(store.has_tensor("model.norm.weight"));
    assert_eq!(store.tensor_f32("model.norm.weight").unwrap(), values);
}

#[test]
fn pyxis_to_gguf_name_mapping() {
    assert_eq!(
        pyxis::gguf::pyxis_to_gguf_name("model.embed_tokens.weight"),
        "token_embd.weight"
    );
    assert_eq!(
        pyxis::gguf::pyxis_to_gguf_name("model.layers.3.self_attn.q_proj.weight"),
        "blk.3.attn_q.weight"
    );
    assert_eq!(
        pyxis::gguf::pyxis_to_gguf_name("model.layers.3.mlp.down_proj.weight"),
        "blk.3.ffn_down.weight"
    );
}

#[test]
fn gguf_to_pyxis_name_mapping() {
    assert_eq!(
        pyxis::gguf::gguf_to_pyxis_name("token_embd.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(
        pyxis::gguf::gguf_to_pyxis_name("blk.12.attn_output.weight"),
        "model.layers.12.self_attn.o_proj.weight"
    );
    assert_eq!(
        pyxis::gguf::gguf_to_pyxis_name("blk.12.ffn_gate.weight"),
        "model.layers.12.mlp.gate_proj.weight"
    );
}

#[test]
fn tensor_f32_from_gguf() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    let values = gguf
        .tensor_f32("output_norm.weight")
        .expect("read output norm tensor");

    assert_eq!(values.len(), 1024);
    assert!(!values.is_empty());
}

#[test]
fn tensor_f32_q8_0_from_gguf() {
    let Some(gguf) = load_test_gguf() else {
        return;
    };

    let info = gguf
        .tensor_info("token_embd.weight")
        .expect("token embedding tensor info");
    let expected_len: usize = info.dims.iter().product::<u64>() as usize;

    let values = gguf
        .tensor_f32("token_embd.weight")
        .expect("read q8 token embedding tensor");

    assert_eq!(values.len(), expected_len);
}

#[test]
fn has_tensor_gguf() {
    let path = Path::new(GGUF_PATH);
    if !path.exists() {
        return;
    }

    let store = TensorStore::load(path).expect("load gguf tensor store");

    assert!(store.has_tensor("model.embed_tokens.weight"));
    assert!(store.has_tensor("model.norm.weight"));
    assert!(!store.has_tensor("model.layers.0.missing.weight"));
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
