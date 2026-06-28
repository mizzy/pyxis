use std::path::Path;

use pyxis::tokenizer::PyxisTokenizer;

const TOKENIZER_PATH: &str = "/Users/mizzy/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json";

fn load_test_tokenizer() -> Option<PyxisTokenizer> {
    let tokenizer_path = Path::new(TOKENIZER_PATH);
    if !tokenizer_path.exists() {
        return None;
    }

    Some(PyxisTokenizer::load(tokenizer_path).expect("load tokenizer"))
}

#[test]
fn encode_returns_token_ids() {
    let Some(tokenizer) = load_test_tokenizer() else {
        return;
    };

    let token_ids = tokenizer.encode("Hello");

    assert!(!token_ids.is_empty());
}

#[test]
fn decode_roundtrips_with_encode() {
    let Some(tokenizer) = load_test_tokenizer() else {
        return;
    };
    let input = "Hello, Pyxis!";

    let token_ids = tokenizer.encode(input);
    let output = tokenizer.decode(&token_ids);

    assert_eq!(output, input);
}

#[test]
fn encode_empty_string() {
    let Some(tokenizer) = load_test_tokenizer() else {
        return;
    };

    let token_ids = tokenizer.encode("");

    assert!(token_ids.is_empty());
}

#[test]
fn load_returns_error_for_missing_file() {
    let path = Path::new("/tmp/pyxis-missing-tokenizer.json");

    let result = PyxisTokenizer::load(path);

    assert!(result.is_err());
}
