use metal::*;
use std::ffi::c_void;
use std::mem;

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_features [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= out_features) return;

    float sum = 0.0;
    uint row_start = gid * in_features;
    for (uint j = 0; j < in_features; j++) {
        sum += input[j] * weight[row_start + j];
    }
    output[gid] = sum;
}
"#;

pub struct MetalMatmul {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalMatmul {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
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

        let in_features_u32 =
            u32::try_from(in_features).expect("in_features exceeds Metal shader uint range");
        let out_features_u32 =
            u32::try_from(out_features).expect("out_features exceeds Metal shader uint range");


        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            mem::size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buf = self.device.new_buffer_with_data(
            weight.as_ptr() as *const c_void,
            mem::size_of_val(weight) as u64,
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
        encoder.set_buffer(1, Some(&weight_buf), 0);
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

        let grid_size = MTLSize::new(out_features as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            self.pipeline
                .max_total_threads_per_threadgroup()
                .min(out_features as u64),
            1,
            1,
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = output_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, out_features) }.to_vec()
    }
}
