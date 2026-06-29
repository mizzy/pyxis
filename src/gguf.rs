use crate::weights::Weights;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

const DATA_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    BF16,
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub tensor_type: GgmlType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    mmap: Mmap,
}

impl GgufFile {
    pub fn parse(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != *b"GGUF" {
            return Err(invalid_data("invalid GGUF magic"));
        }

        let version = read_u32(&mut reader)?;
        if version != 3 {
            return Err(invalid_data(format!("unsupported GGUF version: {version}")));
        }

        let n_tensors = read_u64(&mut reader)?;
        let n_kv = read_u64(&mut reader)?;

        let mut metadata = HashMap::with_capacity(u64_to_usize(n_kv, "metadata count")?);
        for _ in 0..n_kv {
            let key = read_string(&mut reader)?;
            let value_type = read_u32(&mut reader)?;
            let value = read_value(&mut reader, value_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(u64_to_usize(n_tensors, "tensor count")?);
        for _ in 0..n_tensors {
            let name = read_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut reader)?);
            }
            let tensor_type = GgmlType::from_u32(read_u32(&mut reader)?)?;
            let offset = read_u64(&mut reader)?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                tensor_type,
                offset,
            });
        }

        let data_offset = align_to(reader.stream_position()?, DATA_ALIGNMENT)?;

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            mmap,
        })
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let info = self.tensor_info(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor not found: {name}"))
        })?;
        let n_elements = tensor_element_count(&info.dims)?;
        let data_start = self.data_offset.checked_add(info.offset).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor data offset overflow")
        })?;

        match info.tensor_type {
            GgmlType::F32 => self.read_f32_tensor(data_start, n_elements),
            GgmlType::F16 => self.read_f16_tensor(data_start, n_elements),
            GgmlType::BF16 => self.read_bf16_tensor(data_start, n_elements),
            GgmlType::Q8_0 => self.read_q8_0_tensor(data_start, n_elements),
        }
    }

    pub fn tensor_weights(&self, name: &str) -> io::Result<Weights> {
        let info = self.tensor_info(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor not found: {name}"))
        })?;
        let n_elements = tensor_element_count(&info.dims)?;
        let data_start = self.data_offset.checked_add(info.offset).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor data offset overflow")
        })?;

        match info.tensor_type {
            GgmlType::BF16 => {
                let byte_len = checked_byte_len(n_elements, 2)?;
                let bytes = self.tensor_bytes(data_start, byte_len)?;
                Ok(Weights::Bf16(
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect(),
                ))
            }
            GgmlType::F32 | GgmlType::F16 | GgmlType::Q8_0 => {
                Ok(Weights::F32(self.tensor_f32(name)?))
            }
        }
    }

    pub fn tensor_f32_by_pyxis_name(&self, pyxis_name: &str) -> io::Result<Vec<f32>> {
        if self.tensor_info(pyxis_name).is_some() {
            return self.tensor_f32(pyxis_name);
        }

        self.tensor_f32(&pyxis_to_gguf_name(pyxis_name))
    }

    pub fn tensor_weights_by_pyxis_name(&self, pyxis_name: &str) -> io::Result<Weights> {
        if self.tensor_info(pyxis_name).is_some() {
            return self.tensor_weights(pyxis_name);
        }

        self.tensor_weights(&pyxis_to_gguf_name(pyxis_name))
    }

    pub fn has_tensor_by_pyxis_name(&self, pyxis_name: &str) -> bool {
        self.tensor_info(pyxis_name).is_some()
            || self.tensor_info(&pyxis_to_gguf_name(pyxis_name)).is_some()
    }

    fn read_f32_tensor(&self, start: u64, n_elements: usize) -> io::Result<Vec<f32>> {
        let byte_len = checked_byte_len(n_elements, 4)?;
        let bytes = self.tensor_bytes(start, byte_len)?;

        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn read_f16_tensor(&self, start: u64, n_elements: usize) -> io::Result<Vec<f32>> {
        let byte_len = checked_byte_len(n_elements, 2)?;
        let bytes = self.tensor_bytes(start, byte_len)?;

        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect())
    }

    fn read_bf16_tensor(&self, start: u64, n_elements: usize) -> io::Result<Vec<f32>> {
        let byte_len = checked_byte_len(n_elements, 2)?;
        let bytes = self.tensor_bytes(start, byte_len)?;

        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect())
    }

    fn read_q8_0_tensor(&self, start: u64, n_elements: usize) -> io::Result<Vec<f32>> {
        let block_size = 32;
        let bytes_per_block = 34;
        let n_blocks = n_elements.div_ceil(block_size);
        let byte_len = checked_byte_len(n_blocks, bytes_per_block)?;
        let bytes = self.tensor_bytes(start, byte_len)?;

        let mut result = Vec::with_capacity(n_elements);
        for block in bytes.chunks_exact(bytes_per_block) {
            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for &value in &block[2..] {
                result.push(value as i8 as f32 * scale);
            }
        }
        result.truncate(n_elements);
        Ok(result)
    }

    fn tensor_bytes(&self, start: u64, byte_len: usize) -> io::Result<&[u8]> {
        let start = u64_to_usize(start, "tensor start offset")?;
        let end = start.checked_add(byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor end offset overflow")
        })?;

        if end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "tensor data extends past end of file",
            ));
        }

        Ok(&self.mmap[start..end])
    }
}

pub fn gguf_to_pyxis_name(gguf_name: &str) -> String {
    match gguf_name {
        "token_embd.weight" => "model.embed_tokens.weight".to_string(),
        "output_norm.weight" => "model.norm.weight".to_string(),
        "output.weight" => "lm_head.weight".to_string(),
        _ => {
            if let Some(rest) = gguf_name.strip_prefix("blk.")
                && let Some((n, suffix)) = rest.split_once('.')
            {
                let mapped_suffix = match suffix {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "attn_norm.weight" => "input_layernorm.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    other => other,
                };
                return format!("model.layers.{n}.{mapped_suffix}");
            }
            gguf_name.to_string()
        }
    }
}

pub fn pyxis_to_gguf_name(pyxis_name: &str) -> String {
    match pyxis_name {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        "lm_head.weight" => "output.weight".to_string(),
        _ => {
            if let Some(rest) = pyxis_name.strip_prefix("model.layers.")
                && let Some((n, suffix)) = rest.split_once('.')
            {
                let mapped_suffix = match suffix {
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "self_attn.q_norm.weight" => "attn_q_norm.weight",
                    "self_attn.k_norm.weight" => "attn_k_norm.weight",
                    "input_layernorm.weight" => "attn_norm.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    other => other,
                };
                return format!("blk.{n}.{mapped_suffix}");
            }
            pyxis_name.to_string()
        }
    }
}

impl GgmlType {
    fn from_u32(value: u32) -> io::Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            28 => Ok(Self::BF16),
            _ => Err(invalid_data(format!(
                "unsupported GGML tensor type: {value}"
            ))),
        }
    }
}

fn read_value(reader: &mut impl Read, value_type: u32) -> io::Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::U8(read_u8(reader)?)),
        1 => Ok(GgufValue::I8(read_i8(reader)?)),
        2 => Ok(GgufValue::U16(read_u16(reader)?)),
        3 => Ok(GgufValue::I16(read_i16(reader)?)),
        4 => Ok(GgufValue::U32(read_u32(reader)?)),
        5 => Ok(GgufValue::I32(read_i32(reader)?)),
        6 => Ok(GgufValue::F32(read_f32(reader)?)),
        7 => read_bool(reader),
        8 => Ok(GgufValue::String(read_string(reader)?)),
        9 => read_array(reader),
        10 => Ok(GgufValue::U64(read_u64(reader)?)),
        11 => Ok(GgufValue::I64(read_i64(reader)?)),
        12 => Ok(GgufValue::F64(read_f64(reader)?)),
        _ => Err(invalid_data(format!(
            "unsupported GGUF value type: {value_type}"
        ))),
    }
}

fn read_bool(reader: &mut impl Read) -> io::Result<GgufValue> {
    match read_u8(reader)? {
        0 => Ok(GgufValue::Bool(false)),
        1 => Ok(GgufValue::Bool(true)),
        value => Err(invalid_data(format!("invalid bool value: {value}"))),
    }
}

fn read_array(reader: &mut impl Read) -> io::Result<GgufValue> {
    let element_type = read_u32(reader)?;
    let count = u64_to_usize(read_u64(reader)?, "array count")?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_value(reader, element_type)?);
    }
    Ok(GgufValue::Array(values))
}

fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let len = u64_to_usize(read_u64(reader)?, "string length")?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    Ok(read_array_bytes::<1>(reader)?[0])
}

fn read_i8(reader: &mut impl Read) -> io::Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_u16(reader: &mut impl Read) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_array_bytes(reader)?))
}

fn read_i16(reader: &mut impl Read) -> io::Result<i16> {
    Ok(i16::from_le_bytes(read_array_bytes(reader)?))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_array_bytes(reader)?))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_array_bytes(reader)?))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_array_bytes(reader)?))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_array_bytes(reader)?))
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_array_bytes(reader)?))
}

fn read_f64(reader: &mut impl Read) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_array_bytes(reader)?))
}

fn read_array_bytes<const N: usize>(reader: &mut impl Read) -> io::Result<[u8; N]> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn align_to(value: u64, alignment: u64) -> io::Result<u64> {
    let mask = alignment - 1;
    let value = value
        .checked_add(mask)
        .ok_or_else(|| invalid_data("aligned offset overflow"))?;
    Ok(value & !mask)
}

fn u64_to_usize(value: u64, field: &str) -> io::Result<usize> {
    usize::try_from(value).map_err(|_| invalid_data(format!("{field} does not fit in usize")))
}

fn tensor_element_count(dims: &[u64]) -> io::Result<usize> {
    let count = dims.iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| invalid_data("tensor element count overflow"))
    })?;
    u64_to_usize(count, "tensor element count")
}

fn checked_byte_len(n_elements: usize, element_size: usize) -> io::Result<usize> {
    n_elements
        .checked_mul(element_size)
        .ok_or_else(|| invalid_data("tensor byte length overflow"))
}

fn invalid_data(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
