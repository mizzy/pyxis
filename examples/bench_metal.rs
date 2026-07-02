#[cfg(target_os = "macos")]
use pyxis::metal_matmul::MetalMatmul;
#[cfg(target_os = "macos")]
use pyxis::weights::Weights;
#[cfg(target_os = "macos")]
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::hint::black_box;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
const WARMUP_ITERS: usize = 5;
#[cfg(target_os = "macos")]
const BENCH_ITERS: usize = 50;

#[cfg(target_os = "macos")]
fn main() {
    let Some(metal) = MetalMatmul::new() else {
        eprintln!("Metal not available");
        return;
    };

    println!(
        "{:<14} {:<8} {:>8} {:>12}",
        "shape", "kernel", "iters", "us/call"
    );
    println!("{:-<48}", "");

    for (out_features, in_features) in [(2048, 2048), (2048, 6144)] {
        run_shape(&metal, out_features, in_features);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bench_metal requires macOS");
}

#[cfg(target_os = "macos")]
fn run_shape(metal: &MetalMatmul, out_features: usize, in_features: usize) {
    let mut seed = (out_features as u64) << 32 | in_features as u64;
    let input = random_values(in_features, &mut seed);
    let weight = random_values(out_features * in_features, &mut seed);

    let f32_buffer = metal.create_buffer(&weight);
    let f32_us =
        bench_kernel(|| metal.matmul_with_buffer(&input, &f32_buffer, out_features, in_features));
    print_result(out_features, in_features, "f32", f32_us);

    let weight_bf16: Vec<u16> = weight.iter().copied().map(f32_to_bf16).collect();
    let bf16_buffer = metal.create_buffer_raw(
        weight_bf16.as_ptr() as *const c_void,
        std::mem::size_of_val(weight_bf16.as_slice()),
    );
    let bf16_us = bench_kernel(|| {
        metal.matmul_bf16_with_buffer(&input, &bf16_buffer, out_features, in_features)
    });
    print_result(out_features, in_features, "bf16", bf16_us);

    let q8_weights = Weights::quantize_int8(&weight, 32);
    let Weights::Int8 {
        data,
        scales,
        block_size,
        ..
    } = &q8_weights
    else {
        panic!("expected int8 weights");
    };
    let q8_buffer = metal.create_buffer_raw(data.as_ptr() as *const c_void, data.len());
    let scales_buffer = metal.create_buffer(scales);
    let q8_us = bench_kernel(|| {
        metal.matmul_q8_with_buffer(
            &input,
            &q8_buffer,
            &scales_buffer,
            *block_size,
            out_features,
            in_features,
        )
    });
    print_result(out_features, in_features, "q8", q8_us);
}

#[cfg(target_os = "macos")]
fn bench_kernel<F>(mut kernel: F) -> f64
where
    F: FnMut() -> Vec<f32>,
{
    for _ in 0..WARMUP_ITERS {
        black_box(kernel());
    }

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        black_box(kernel());
    }

    start.elapsed().as_secs_f64() * 1_000_000.0 / BENCH_ITERS as f64
}

#[cfg(target_os = "macos")]
fn print_result(out_features: usize, in_features: usize, kernel: &str, us_per_call: f64) {
    println!(
        "{:<14} {:<8} {:>8} {:>12.2}",
        format!("{out_features}x{in_features}"),
        kernel,
        BENCH_ITERS,
        us_per_call
    );
}

#[cfg(target_os = "macos")]
fn random_values(len: usize, seed: &mut u64) -> Vec<f32> {
    (0..len)
        .map(|_| next_random_f32(seed) * 2.0 - 1.0)
        .collect()
}

#[cfg(target_os = "macos")]
fn next_random_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 40) as u32) as f32 / (1u32 << 24) as f32
}

#[cfg(target_os = "macos")]
fn f32_to_bf16(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}
