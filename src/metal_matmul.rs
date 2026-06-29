use metal::*;
use std::ffi::c_void;
use std::mem;

const THREADS_PER_ROW: u64 = 256;
const INPUT_TILE_ELEMENTS: u64 = 4096;

fn shader_source() -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_ROW = {threads_per_row};
constant uint SIMD_WIDTH = 32;
constant uint INPUT_TILE_ELEMENTS = {input_tile_elements};
constant uint SIMD_GROUPS_PER_ROW = THREADS_PER_ROW / SIMD_WIDTH;

static inline float4 load_float4(device const float* values, uint offset) {{
    packed_float4 packed = *(reinterpret_cast<device const packed_float4*>(values + offset));
    return float4(packed);
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
"#,
        threads_per_row = THREADS_PER_ROW,
        input_tile_elements = INPUT_TILE_ELEMENTS,
    )
}

pub struct MetalMatmul {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalMatmul {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();
        let shader_source = shader_source();
        let library = device
            .new_library_with_source(&shader_source, &CompileOptions::new())
            .ok()?;
        let function = library.get_function("matmul_f32", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;

        Some(Self {
            device,
            command_queue,
            pipeline,
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
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            mem::size_of_val(data) as u64,
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
        assert_eq!(input.len(), in_features);
        assert_eq!(
            weight_buffer.length(),
            (out_features * in_features * mem::size_of::<f32>()) as u64
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

        encoder.set_compute_pipeline_state(&self.pipeline);
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
        encoder
            .set_threadgroup_memory_length(0, INPUT_TILE_ELEMENTS * mem::size_of::<f32>() as u64);

        assert!(
            self.pipeline.max_total_threads_per_threadgroup() >= THREADS_PER_ROW,
            "Metal device does not support {THREADS_PER_ROW} threads per threadgroup"
        );
        let grid_size = MTLSize::new(THREADS_PER_ROW, out_features as u64, 1);
        let threadgroup_size = MTLSize::new(THREADS_PER_ROW, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = output_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, out_features) }.to_vec()
    }
}
