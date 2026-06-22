mod display;
mod safetensors;

use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: pyxis <model.safetensors>");
    let tensors =
        safetensors::parse_header(Path::new(&path)).expect("Failed to parse safetensors header");

    print!("{}", display::format_tensor_table(&tensors));
}
