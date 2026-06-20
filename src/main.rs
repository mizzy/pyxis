mod safetensors;

use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: pyxis <model.safetensors>");
    let tensors = safetensors::parse_header(Path::new(&path)).expect("Failed to parse safetensors header");

    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    println!("{:<60} {:>8} {:>20} {:>24}", "Tensor", "Dtype", "Shape", "Offsets");
    println!("{}", "-".repeat(114));
    for name in names {
        let info = &tensors[name];
        let [start, end] = info.data_offsets;
        println!(
            "{:<60} {:>8} {:>20?} {:>12}..{}",
            name, info.dtype, info.shape, start, end
        );
    }
    println!("\nTotal: {} tensors", tensors.len());
}
