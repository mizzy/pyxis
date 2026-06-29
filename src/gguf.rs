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

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

impl GgufFile {
    pub fn parse(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
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

fn invalid_data(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
