use crate::gguf::GgufFile;
use crate::weights::Weights;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    pub fn element_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: serde_json::Value,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug)]
pub struct SafeTensors {
    mmap: memmap2::Mmap,
    data_offset: usize,
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Deserialize)]
struct SafeTensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ShardedSafeTensors {
    shards: Vec<SafeTensors>,
    tensor_to_shard: HashMap<String, usize>,
}

impl ShardedSafeTensors {
    pub fn load(index_path: &Path) -> io::Result<Self> {
        let json = fs::read_to_string(index_path)?;
        let index: SafeTensorsIndex = serde_json::from_str(&json)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let model_dir = index_path.parent().unwrap_or_else(|| Path::new(""));

        let mut shards = Vec::new();
        let mut shard_to_index = HashMap::new();
        let mut tensor_to_shard = HashMap::new();

        for (tensor_name, shard_file) in index.weight_map {
            let shard_index = if let Some(shard_index) = shard_to_index.get(&shard_file) {
                *shard_index
            } else {
                let shard_path = model_dir.join(&shard_file);
                let shard_index = shards.len();
                shards.push(SafeTensors::load(&shard_path)?);
                shard_to_index.insert(shard_file, shard_index);
                shard_index
            };
            tensor_to_shard.insert(tensor_name, shard_index);
        }

        Ok(Self {
            shards,
            tensor_to_shard,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let shard_index = self.tensor_to_shard.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor not found: {name}"))
        })?;
        self.shards[*shard_index].tensor_f32(name)
    }

    pub fn tensor_weights(&self, name: &str) -> io::Result<Weights> {
        let shard_index = self.tensor_to_shard.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor not found: {name}"))
        })?;
        self.shards[*shard_index].tensor_weights(name)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }
}

#[derive(Debug)]
pub enum TensorStore {
    Single(SafeTensors),
    Sharded(ShardedSafeTensors),
    Gguf(GgufFile),
}

impl TensorStore {
    pub fn load(model_dir: &Path) -> io::Result<Self> {
        if model_dir
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return Ok(Self::Gguf(GgufFile::parse(model_dir)?));
        }

        let single_path = model_dir.join("model.safetensors");
        let index_path = model_dir.join("model.safetensors.index.json");

        if single_path.exists() {
            return Ok(Self::Single(SafeTensors::load(&single_path)?));
        }

        if index_path.exists() {
            return Ok(Self::Sharded(ShardedSafeTensors::load(&index_path)?));
        }

        let mut candidates = Vec::new();
        for entry in fs::read_dir(model_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
                candidates.push(path);
            }
        }
        candidates.sort();

        let path = candidates
            .into_iter()
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no safetensors file found"))?;

        Ok(Self::Single(SafeTensors::load(&path)?))
    }

    pub fn tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        match self {
            Self::Single(safetensors) => safetensors.tensor_f32(name),
            Self::Sharded(safetensors) => safetensors.tensor_f32(name),
            Self::Gguf(gguf) => gguf.tensor_f32_by_pyxis_name(name),
        }
    }

    pub fn tensor_weights(&self, name: &str) -> io::Result<Weights> {
        match self {
            Self::Single(safetensors) => safetensors.tensor_weights(name),
            Self::Sharded(safetensors) => safetensors.tensor_weights(name),
            Self::Gguf(gguf) => gguf.tensor_weights_by_pyxis_name(name),
        }
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        match self {
            Self::Single(safetensors) => safetensors.has_tensor(name),
            Self::Sharded(safetensors) => safetensors.has_tensor(name),
            Self::Gguf(gguf) => gguf.has_tensor_by_pyxis_name(name),
        }
    }

    pub fn tensors(&self) -> HashMap<String, TensorInfo> {
        match self {
            Self::Single(safetensors) => safetensors.tensors().clone(),
            Self::Sharded(sharded) => {
                let mut all = HashMap::new();
                for shard in &sharded.shards {
                    all.extend(shard.tensors().clone());
                }
                all
            }
            Self::Gguf(_) => HashMap::new(),
        }
    }
}

impl SafeTensors {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "safetensors file is too short to contain a header size",
            ));
        }

        let mut size_buf = [0u8; 8];
        size_buf.copy_from_slice(&mmap[..8]);
        let header_size = u64::from_le_bytes(size_buf) as usize;
        let data_offset = 8usize
            .checked_add(header_size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "header size overflow"))?;

        if mmap.len() < data_offset {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "safetensors file is too short to contain the declared header",
            ));
        }

        let header_json: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_offset])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut tensors = HashMap::new();
        for (key, value) in header_json {
            if key == "__metadata__" {
                continue;
            }
            let raw: RawTensorInfo = serde_json::from_value(value)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let dtype = match serde_json::from_value(raw.dtype) {
                Ok(dtype) => dtype,
                Err(_) => continue,
            };
            let info = TensorInfo {
                dtype,
                shape: raw.shape,
                data_offsets: raw.data_offsets,
            };
            tensors.insert(key, info);
        }

        Ok(Self {
            mmap,
            data_offset,
            tensors,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let (info, bytes) = self.tensor_data(name)?;

        Ok(decode_f32(info.dtype, bytes))
    }

    pub fn tensor_weights(&self, name: &str) -> io::Result<Weights> {
        let (info, bytes) = self.tensor_data(name)?;

        match info.dtype {
            Dtype::BF16 => Ok(Weights::Bf16(
                bytes
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect(),
            )),
            Dtype::F32 | Dtype::F16 => Ok(Weights::F32(decode_f32(info.dtype, bytes))),
        }
    }

    fn tensor_data(&self, name: &str) -> io::Result<(&TensorInfo, &[u8])> {
        let info = self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor not found: {name}"))
        })?;

        let [start, end] = info.data_offsets;
        let start = self.data_offset.checked_add(start).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor start offset overflow")
        })?;
        let end = self.data_offset.checked_add(end).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor end offset overflow")
        })?;

        if start > end || end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor data offsets are outside the file",
            ));
        }

        let bytes = &self.mmap[start..end];
        if !bytes.len().is_multiple_of(info.dtype.element_size()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor byte length is not aligned to dtype element size",
            ));
        }

        Ok((info, bytes))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }
}

fn decode_f32(dtype: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors(tensors: serde_json::Value, data: &[u8]) -> NamedTempFile {
        let header_bytes = serde_json::to_vec(&tensors).unwrap();
        let header_size = (header_bytes.len() as u64).to_le_bytes();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&header_size).unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
        file
    }

    fn write_test_safetensors(path: &Path, tensors: serde_json::Value, data: &[u8]) {
        let header_bytes = serde_json::to_vec(&tensors).unwrap();
        let header_size = (header_bytes.len() as u64).to_le_bytes();

        let mut file = File::create(path).unwrap();
        file.write_all(&header_size).unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn sharded_index_deserializes() {
        let index = serde_json::from_str::<SafeTensorsIndex>(
            r#"{
                "metadata": {"total_size": 24},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();

        assert_eq!(index.weight_map.len(), 2);
        assert_eq!(
            index.weight_map["model.embed_tokens.weight"],
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(
            index.weight_map["lm_head.weight"],
            "model-00002-of-00002.safetensors"
        );
    }

    #[test]
    fn tensor_store_loads_single_file() {
        let dir = tempfile::tempdir().unwrap();
        write_test_safetensors(
            &dir.path().join("model.safetensors"),
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8]
                }
            }),
            &f32_bytes(&[1.0, 2.0]),
        );

        let store = TensorStore::load(dir.path()).unwrap();

        assert!(matches!(store, TensorStore::Single(_)));
        assert_eq!(store.tensor_f32("values").unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn tensor_store_loads_sharded() {
        let dir = tempfile::tempdir().unwrap();
        let shard_1 = "model-00001-of-00002.safetensors";
        let shard_2 = "model-00002-of-00002.safetensors";

        write_test_safetensors(
            &dir.path().join(shard_1),
            serde_json::json!({
                "first.weight": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8]
                },
                "shared.weight": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [8, 12]
                }
            }),
            &f32_bytes(&[1.0, 2.0, 9.0]),
        );
        write_test_safetensors(
            &dir.path().join(shard_2),
            serde_json::json!({
                "second.weight": {
                    "dtype": "F32",
                    "shape": [3],
                    "data_offsets": [0, 12]
                }
            }),
            &f32_bytes(&[3.0, 4.0, 5.0]),
        );
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {"total_size": 20},
                "weight_map": {
                    "first.weight": shard_1,
                    "shared.weight": shard_1,
                    "second.weight": shard_2
                }
            })
            .to_string(),
        )
        .unwrap();

        let store = TensorStore::load(dir.path()).unwrap();

        let TensorStore::Sharded(sharded) = &store else {
            panic!("expected sharded tensor store");
        };
        assert_eq!(sharded.shards.len(), 2);
        assert_eq!(store.tensor_f32("first.weight").unwrap(), vec![1.0, 2.0]);
        assert_eq!(store.tensor_f32("shared.weight").unwrap(), vec![9.0]);
        assert_eq!(
            store.tensor_f32("second.weight").unwrap(),
            vec![3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn tensor_store_errors_on_empty_dir() {
        let dir = tempfile::tempdir().unwrap();

        let error = TensorStore::load(dir.path()).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("no safetensors file found"));
    }

    #[test]
    fn dtype_element_size_returns_byte_widths() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::F16.element_size(), 2);
        assert_eq!(Dtype::BF16.element_size(), 2);
    }

    #[test]
    fn dtype_deserializes_from_json_strings() {
        assert_eq!(
            serde_json::from_str::<Dtype>("\"F32\"").unwrap(),
            Dtype::F32
        );
        assert_eq!(
            serde_json::from_str::<Dtype>("\"F16\"").unwrap(),
            Dtype::F16
        );
        assert_eq!(
            serde_json::from_str::<Dtype>("\"BF16\"").unwrap(),
            Dtype::BF16
        );
    }

    #[test]
    fn load_returns_tensor_metadata() {
        let file = create_test_safetensors(
            serde_json::json!({
                "__metadata__": {"format": "pt"},
                "weight": {
                    "dtype": "F32",
                    "shape": [3, 4],
                    "data_offsets": [0, 48]
                },
                "bias": {
                    "dtype": "BF16",
                    "shape": [4],
                    "data_offsets": [48, 56]
                }
            }),
            &[0u8; 56],
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensor_names().len(), 2);
        assert!(tensors.tensor_info("__metadata__").is_none());

        let weight = tensors.tensor_info("weight").unwrap();
        assert_eq!(weight.dtype, Dtype::F32);
        assert_eq!(weight.shape, vec![3, 4]);
        assert_eq!(weight.data_offsets, [0, 48]);

        let bias = tensors.tensor_info("bias").unwrap();
        assert_eq!(bias.dtype, Dtype::BF16);
        assert_eq!(bias.shape, vec![4]);
        assert_eq!(bias.data_offsets, [48, 56]);
    }

    #[test]
    fn has_tensor_returns_true_for_existing() {
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }
            }),
            &1.0f32.to_le_bytes(),
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert!(tensors.has_tensor("values"));
    }

    #[test]
    fn has_tensor_returns_false_for_missing() {
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }
            }),
            &1.0f32.to_le_bytes(),
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert!(!tensors.has_tensor("missing"));
    }

    #[test]
    fn load_skips_tensor_with_unsupported_dtype() {
        let data: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .chain([1i64, 2].iter().flat_map(|value| value.to_le_bytes()))
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "float_values": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8]
                },
                "int_values": {
                    "dtype": "I64",
                    "shape": [2],
                    "data_offsets": [8, 24]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensor_f32("float_values").unwrap(), vec![1.0, 2.0]);
        assert!(tensors.tensor_info("int_values").is_none());
    }

    #[test]
    fn load_returns_error_for_nonexistent_file() {
        let err = SafeTensors::load(Path::new("/tmp/pyxis-missing-file.safetensors")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn tensor_f32_reads_f32_tensor_data() {
        let values = [1.0f32, 2.0, 3.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [3],
                    "data_offsets": [0, 12]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensor_f32("values").unwrap(), values);
    }

    #[test]
    fn tensor_f32_reads_bf16_tensor_data() {
        let values = [1.5f32, -2.25, 3.5];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "BF16",
                    "shape": [3],
                    "data_offsets": [0, 6]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let loaded = tensors.tensor_f32("values").unwrap();

        assert_eq!(loaded.len(), values.len());
        for (actual, expected) in loaded.iter().zip(values) {
            assert!((actual - expected).abs() < 0.01);
        }
    }

    #[test]
    fn tensor_weights_returns_bf16_for_bf16_dtype() {
        let values = [1.5f32, -2.25, 3.5];
        let raw_bf16: Vec<u16> = values
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_bits())
            .collect();
        let data: Vec<u8> = raw_bf16
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "BF16",
                    "shape": [3],
                    "data_offsets": [0, 6]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let weights = tensors.tensor_weights("values").unwrap();

        assert!(matches!(weights, crate::weights::Weights::Bf16(values) if values == raw_bf16));
    }

    #[test]
    fn tensor_f32_returns_error_for_unknown_tensor() {
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }
            }),
            &1.0f32.to_le_bytes(),
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let err = tensors.tensor_f32("missing").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn tensors_returns_all_metadata() {
        let file = create_test_safetensors(
            serde_json::json!({
                "weight": {
                    "dtype": "F32",
                    "shape": [3, 4],
                    "data_offsets": [0, 48]
                }
            }),
            &[0u8; 48],
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensors().len(), 1);
        assert!(tensors.tensors().contains_key("weight"));
    }

    #[test]
    fn tensor_f32_reads_f16_tensor_data() {
        let values = [1.5f32, -2.25, 3.5];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F16",
                    "shape": [3],
                    "data_offsets": [0, 6]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let loaded = tensors.tensor_f32("values").unwrap();

        assert_eq!(loaded.len(), values.len());
        for (actual, expected) in loaded.iter().zip(values) {
            assert!((actual - expected).abs() < 0.001);
        }
    }

    #[test]
    fn load_ignores_metadata_entry() {
        let file = create_test_safetensors(
            serde_json::json!({
            "__metadata__": {"format": "pt"},
            "weight": {
                "dtype": "F32",
                "shape": [3, 4],
                "data_offsets": [0, 48]
            }
            }),
            &[0u8; 48],
        );
        let tensors = SafeTensors::load(file.path()).unwrap();
        assert!(!tensors.tensors().contains_key("__metadata__"));
    }
}
