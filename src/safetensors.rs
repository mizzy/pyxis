use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

pub fn parse_header(path: &Path) -> io::Result<HashMap<String, TensorInfo>> {
    let mut file = File::open(path)?;

    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;

    let header_json: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut tensors = HashMap::new();
    for (key, value) in header_json {
        if key == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        tensors.insert(key, info);
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors() -> NamedTempFile {
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "weight": {
                "dtype": "F32",
                "shape": [3, 4],
                "data_offsets": [0, 48]
            },
            "bias": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [48, 64]
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_size = (header_bytes.len() as u64).to_le_bytes();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&header_size).unwrap();
        file.write_all(&header_bytes).unwrap();
        // Write dummy tensor data (64 bytes = 16 f32 values)
        file.write_all(&[0u8; 64]).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn parse_header_returns_tensor_info() {
        let file = create_test_safetensors();
        let tensors = parse_header(file.path()).unwrap();

        assert_eq!(tensors.len(), 2);

        let weight = &tensors["weight"];
        assert_eq!(weight.dtype, "F32");
        assert_eq!(weight.shape, vec![3, 4]);
        assert_eq!(weight.data_offsets, [0, 48]);

        let bias = &tensors["bias"];
        assert_eq!(bias.dtype, "F32");
        assert_eq!(bias.shape, vec![4]);
        assert_eq!(bias.data_offsets, [48, 64]);
    }

    #[test]
    fn parse_header_skips_metadata() {
        let file = create_test_safetensors();
        let tensors = parse_header(file.path()).unwrap();
        assert!(!tensors.contains_key("__metadata__"));
    }
}
