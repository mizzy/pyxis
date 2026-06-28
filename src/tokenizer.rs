use std::io;
use std::path::Path;

pub struct PyxisTokenizer {
    inner: tokenizers::Tokenizer,
}

impl PyxisTokenizer {
    pub fn load(path: &Path) -> io::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(io::Error::other)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text, false).unwrap().get_ids().to_vec()
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.inner.decode(token_ids, true).unwrap()
    }
}
