use metal::*;
use std::ffi::c_void;
use std::mem;

const THREADS_PER_ROW: u64 = 256;
const SIMD_WIDTH: u64 = 32;
const SIMD_GROUPS_PER_THREADGROUP: u64 = THREADS_PER_ROW / SIMD_WIDTH;
const INPUT_TILE_ELEMENTS: u64 = 4096;

fn shader_source() -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_ROW = {threads_per_row};
constant uint SIMD_WIDTH = {simd_width};
constant uint INPUT_TILE_ELEMENTS = {input_tile_elements};
constant uint SIMD_GROUPS_PER_ROW = THREADS_PER_ROW / SIMD_WIDTH;

static inline float4 load_float4(device const float* values, uint offset) {{
    packed_float4 packed = *(reinterpret_cast<device const packed_float4*>(values + offset));
    return float4(packed);
}}

static inline char4 load_char4(device const char* values, uint offset) {{
    packed_char4 packed = *(reinterpret_cast<device const packed_char4*>(values + offset));
    return char4(packed);
}}

static inline float bf16_to_f32(ushort value) {{
    return as_type<float>(uint(value) << 16);
}}

kernel void matmul_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_features [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{{
    uint row = gid.y;
    if (row >= out_features) return;

    uint local_id = tid.x;
    uint row_start = row * in_features;
    threadgroup float partial_sums[SIMD_GROUPS_PER_ROW];

    float sum = 0.0;

    for (uint tile_base = 0; tile_base < in_features; tile_base += INPUT_TILE_ELEMENTS) {{
        uint tile_len = min(INPUT_TILE_ELEMENTS, in_features - tile_base);
        uint vec4_count = tile_len / 4;

        for (uint vec4_index = local_id; vec4_index < vec4_count; vec4_index += THREADS_PER_ROW) {{
            uint offset = vec4_index * 4;
            float4 input_vec = load_float4(input + tile_base, offset);
            shared_input[offset] = input_vec.x;
            shared_input[offset + 1] = input_vec.y;
            shared_input[offset + 2] = input_vec.z;
            shared_input[offset + 3] = input_vec.w;
        }}

        for (uint offset = vec4_count * 4 + local_id; offset < tile_len; offset += THREADS_PER_ROW) {{
            shared_input[offset] = input[tile_base + offset];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint row_tile_start = row_start + tile_base;
        for (uint vec4_index = local_id; vec4_index < vec4_count; vec4_index += THREADS_PER_ROW) {{
            uint offset = vec4_index * 4;
            float4 input_vec = float4(
                shared_input[offset],
                shared_input[offset + 1],
                shared_input[offset + 2],
                shared_input[offset + 3]
            );
            float4 weight_vec = load_float4(weight + row_tile_start, offset);
            sum += dot(input_vec, weight_vec);
        }}

        for (uint offset = vec4_count * 4 + local_id; offset < tile_len; offset += THREADS_PER_ROW) {{
            sum += shared_input[offset] * weight[row_tile_start + offset];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    sum = simd_sum(sum);

    if (simd_lane == 0) {{
        partial_sums[simd_group] = sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {{
        float total = 0.0;
        for (uint i = 0; i < SIMD_GROUPS_PER_ROW; i++) {{
            total += partial_sums[i];
        }}
        output[row] = total;
    }}
}}

kernel void matmul_bf16(
    device const float* input [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_features [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{{
    uint row = gid.y;
    if (row >= out_features) return;

    uint local_id = tid.x;
    uint row_start = row * in_features;
    threadgroup float partial_sums[SIMD_GROUPS_PER_ROW];

    float sum = 0.0;

    for (uint tile_base = 0; tile_base < in_features; tile_base += INPUT_TILE_ELEMENTS) {{
        uint tile_len = min(INPUT_TILE_ELEMENTS, in_features - tile_base);
        uint vec4_count = tile_len / 4;

        for (uint vec4_index = local_id; vec4_index < vec4_count; vec4_index += THREADS_PER_ROW) {{
            uint offset = vec4_index * 4;
            float4 input_vec = load_float4(input + tile_base, offset);
            shared_input[offset] = input_vec.x;
            shared_input[offset + 1] = input_vec.y;
            shared_input[offset + 2] = input_vec.z;
            shared_input[offset + 3] = input_vec.w;
        }}

        for (uint offset = vec4_count * 4 + local_id; offset < tile_len; offset += THREADS_PER_ROW) {{
            shared_input[offset] = input[tile_base + offset];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint row_tile_start = row_start + tile_base;
        for (uint vec4_index = local_id; vec4_index < vec4_count; vec4_index += THREADS_PER_ROW) {{
            uint offset = vec4_index * 4;
            float4 input_vec = float4(
                shared_input[offset],
                shared_input[offset + 1],
                shared_input[offset + 2],
                shared_input[offset + 3]
            );
            uint w_offset = row_tile_start + offset;
            float4 weight_vec = float4(
                bf16_to_f32(weight[w_offset]),
                bf16_to_f32(weight[w_offset + 1]),
                bf16_to_f32(weight[w_offset + 2]),
                bf16_to_f32(weight[w_offset + 3])
            );
            sum += dot(input_vec, weight_vec);
        }}

        for (uint offset = vec4_count * 4 + local_id; offset < tile_len; offset += THREADS_PER_ROW) {{
            sum += shared_input[offset] * bf16_to_f32(weight[row_tile_start + offset]);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    sum = simd_sum(sum);

    if (simd_lane == 0) {{
        partial_sums[simd_group] = sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {{
        float total = 0.0;
        for (uint i = 0; i < SIMD_GROUPS_PER_ROW; i++) {{
            total += partial_sums[i];
        }}
        output[row] = total;
    }}
}}

kernel void matmul_q8(
    device const float* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_features [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    device const float* scales [[buffer(5)]],
    constant uint& block_size [[buffer(6)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{{
    uint row = gid.y * SIMD_GROUPS_PER_ROW + simd_group;
    if (row >= out_features) return;

    uint blocks_per_row = in_features / block_size;
    uint row_start = row * in_features;
    uint row_block_start = row * blocks_per_row;

    float sum = 0.0;

    if (blocks_per_row == 1) {{
        uint vec4_count = in_features / 4;
        for (uint v = simd_lane; v < vec4_count; v += SIMD_WIDTH) {{
            uint o = v * 4;
            float4 in_vec = load_float4(input, o);
            char4 w = load_char4(weight, row_start + o);
            sum += dot(in_vec, float4(w));
        }}
        for (uint o = vec4_count * 4 + simd_lane; o < in_features; o += SIMD_WIDTH) {{
            sum += input[o] * (float)(weight[row_start + o]);
        }}
        sum *= scales[row_block_start];
    }} else {{
        uint lane_block = simd_lane / 4;
        uint lane_off = (simd_lane % 4) * 8;
        uint blocks_per_iter = SIMD_WIDTH / 4;
        for (uint ib = lane_block; ib < blocks_per_row; ib += blocks_per_iter) {{
            float scale = scales[row_block_start + ib];
            uint base = ib * block_size + lane_off;
            float4 in0 = load_float4(input + base, 0);
            char4 w0 = load_char4(weight, row_start + base);
            float4 in1 = load_float4(input + base, 4);
            char4 w1 = load_char4(weight, row_start + base + 4);
            sum += (dot(in0, float4(w0)) + dot(in1, float4(w1))) * scale;
        }}
    }}

    sum = simd_sum(sum);

    if (simd_lane == 0) {{
        output[row] = sum;
    }}
}}
"#,
        threads_per_row = THREADS_PER_ROW,
        simd_width = SIMD_WIDTH,
        input_tile_elements = INPUT_TILE_ELEMENTS,
    )
}

#[derive(Clone, Copy)]
struct MatmulShape {
    out_features: usize,
    in_features: usize,
}

#[derive(Clone, Copy)]
enum KernelKind<'a> {
    Dense,
    Q8 { scales: &'a Buffer, block_size: u32 },
}

pub struct MetalMatmul {
    device: Device,
    command_queue: CommandQueue,
    pipeline_f32: ComputePipelineState,
    pipeline_bf16: ComputePipelineState,
    pipeline_q8: ComputePipelineState,
}

impl MetalMatmul {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();
        let shader_source = shader_source();
        let library = device
            .new_library_with_source(&shader_source, &CompileOptions::new())
            .ok()?;
        let function_f32 = library.get_function("matmul_f32", None).ok()?;
        let pipeline_f32 = device
            .new_compute_pipeline_state_with_function(&function_f32)
            .ok()?;
        let function_bf16 = library.get_function("matmul_bf16", None).ok()?;
        let pipeline_bf16 = device
            .new_compute_pipeline_state_with_function(&function_bf16)
            .ok()?;
        let function_q8 = library.get_function("matmul_q8", None).ok()?;
        let pipeline_q8 = device
            .new_compute_pipeline_state_with_function(&function_q8)
            .ok()?;
        if pipeline_q8.thread_execution_width() != SIMD_WIDTH {
            return None;
        }

        Some(Self {
            device,
            command_queue,
            pipeline_f32,
            pipeline_bf16,
            pipeline_q8,
        })
    }

    pub fn matmul(
        &self,
        input: &[f32],
        weight: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        assert_eq!(input.len(), in_features);
        assert_eq!(weight.len(), out_features * in_features);

        if out_features == 0 {
            return Vec::new();
        }

        let weight_buf = self.create_buffer(weight);
        self.matmul_with_buffer(input, &weight_buf, out_features, in_features)
    }

    pub fn create_buffer(&self, data: &[f32]) -> Buffer {
        self.create_buffer_raw(data.as_ptr() as *const c_void, mem::size_of_val(data))
    }

    pub fn create_buffer_raw(&self, ptr: *const c_void, byte_len: usize) -> Buffer {
        self.device.new_buffer_with_data(
            ptr,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn matmul_with_buffer(
        &self,
        input: &[f32],
        weight_buffer: &Buffer,
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        self.matmul_with_pipeline(
            &self.pipeline_f32,
            input,
            weight_buffer,
            MatmulShape {
                out_features,
                in_features,
            },
            mem::size_of::<f32>(),
            KernelKind::Dense,
        )
    }

    pub fn matmul_bf16_with_buffer(
        &self,
        input: &[f32],
        weight_buffer: &Buffer,
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        self.matmul_with_pipeline(
            &self.pipeline_bf16,
            input,
            weight_buffer,
            MatmulShape {
                out_features,
                in_features,
            },
            mem::size_of::<u16>(),
            KernelKind::Dense,
        )
    }

    pub fn matmul_q8_with_buffer(
        &self,
        input: &[f32],
        weight_buffer: &Buffer,
        scales_buffer: &Buffer,
        block_size: usize,
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        assert!(block_size > 0);
        assert!(
            block_size == 32 || block_size == in_features,
            "Q8 Metal kernel supports block_size 32 (GGUF Q8_0) or per-row scales"
        );
        assert!(
            in_features.is_multiple_of(block_size),
            "in_features must be a multiple of block_size for the Q8 Metal kernel"
        );
        assert_eq!(
            scales_buffer.length(),
            ((out_features * in_features).div_ceil(block_size) * mem::size_of::<f32>()) as u64
        );
        let block_size_u32 =
            u32::try_from(block_size).expect("block_size exceeds Metal shader uint range");

        self.matmul_with_pipeline(
            &self.pipeline_q8,
            input,
            weight_buffer,
            MatmulShape {
                out_features,
                in_features,
            },
            mem::size_of::<i8>(),
            KernelKind::Q8 {
                scales: scales_buffer,
                block_size: block_size_u32,
            },
        )
    }

    fn matmul_with_pipeline(
        &self,
        pipeline: &ComputePipelineState,
        input: &[f32],
        weight_buffer: &Buffer,
        shape: MatmulShape,
        weight_element_size: usize,
        kind: KernelKind<'_>,
    ) -> Vec<f32> {
        let MatmulShape {
            out_features,
            in_features,
        } = shape;
        assert_eq!(input.len(), in_features);
        assert_eq!(
            weight_buffer.length(),
            (out_features * in_features * weight_element_size) as u64
        );

        if out_features == 0 {
            return Vec::new();
        }

        let in_features_u32 =
            u32::try_from(in_features).expect("in_features exceeds Metal shader uint range");
        let out_features_u32 =
            u32::try_from(out_features).expect("out_features exceeds Metal shader uint range");

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            mem::size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self.device.new_buffer(
            (out_features * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &in_features_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &out_features_u32 as *const u32 as *const c_void,
        );
        let (rows_per_threadgroup, uses_threadgroup_memory) = match kind {
            KernelKind::Dense => (1, true),
            KernelKind::Q8 { scales, block_size } => {
                encoder.set_buffer(5, Some(scales), 0);
                encoder.set_bytes(
                    6,
                    mem::size_of::<u32>() as u64,
                    &block_size as *const u32 as *const c_void,
                );
                (SIMD_GROUPS_PER_THREADGROUP, false)
            }
        };
        if uses_threadgroup_memory {
            encoder.set_threadgroup_memory_length(
                0,
                INPUT_TILE_ELEMENTS * mem::size_of::<f32>() as u64,
            );
        }

        assert!(
            pipeline.max_total_threads_per_threadgroup() >= THREADS_PER_ROW,
            "Metal device does not support {THREADS_PER_ROW} threads per threadgroup"
        );
        let grid_y = (out_features as u64).div_ceil(rows_per_threadgroup);
        let grid_size = MTLSize::new(THREADS_PER_ROW, grid_y, 1);
        let threadgroup_size = MTLSize::new(THREADS_PER_ROW, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = output_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, out_features) }.to_vec()
    }
}
